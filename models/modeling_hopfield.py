import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaMLP,
)

from .configuration_calm import CALMConfig
from .configuration_autoencoder import AutoencoderConfig
from .modeling_autoencoder import Autoencoder
from .modeling_calm import CALM, CustomCausalLMOutput
from .modeling_energy import MLPGenerator


class HopfieldLayer(nn.Module):
    """
    Hopfield network layer replacing multi-head self-attention.

    Two memory sources:
      - Episodic: causal accumulation of patterns from the current sequence.
        W_j = sum_{i<j} x_i (x) x_i, retrieval y_j = W_j . x_j
        Resets at sequence boundaries.
      - Semantic: persistent Hebbian matrix accumulated across training.
        W_sem = EMA of (sum x_i (x) x_i) from past sequences.
        Consolidation: after each sequence, episodic patterns are deposited
        into W_semantic via exponential moving average.

    Each stored pattern serves as both key and value (no Q/K/V projections).
    Sparse top-k thresholding per query on episodic retrieval.
    Episodic and semantic outputs are combined via a content-based gate.
    """

    def __init__(self, hidden_size, top_k, semantic_momentum=0.99):
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.scale = hidden_size ** -0.5
        self.semantic_momentum = semantic_momentum

        # Persistent Hebbian matrix — accumulated across training, not updated by backprop
        self.register_buffer('W_semantic', torch.zeros(hidden_size, hidden_size))
        self.register_buffer('has_semantic', torch.tensor(False))

        # Temp storage for consolidation after forward pass
        self._pending_states = None
        self._semantic_target_norm = 1.0

        # No output projection — retrieval feeds into the MLP gate instead

    @torch.no_grad()
    def consolidate(self):
        """Deposit pending episodic patterns into semantic memory (EMA update)."""
        if self._pending_states is None:
            return
        patterns = self._pending_states.reshape(-1, self.hidden_size)  # (B*T, D)
        # Mean-center to deposit covariance, not correlation.
        mean = patterns.mean(dim=0, keepdim=True)
        centered = patterns - mean

        if not self.has_semantic:
            W_new = (centered.T @ centered) / centered.shape[0]
            self.W_semantic.copy_(W_new)
            self._semantic_target_norm = W_new.norm().item()
            self.has_semantic.fill_(True)
        else:
            # Per-pattern adaptive blend between raw deposit and replay.
            # Strong retrieval (aligned with dominant directions) -> more raw deposit
            # to dilute amplification. Weak retrieval (orthogonal) -> more replay
            # to reinforce whatever structure exists.
            retrieved = centered @ self.W_semantic  # (N, D)
            # Alignment: how strongly each pattern activates existing structure
            alignment = (retrieved * centered).sum(dim=-1, keepdim=True)  # (N, 1)
            alignment = alignment / (centered.norm(dim=-1, keepdim=True) *
                                     retrieved.norm(dim=-1, keepdim=True) + 1e-8)
            # alpha=1 means fully raw, alpha=0 means fully replay
            alpha = alignment.abs()  # (N, 1), in [0, 1]

            retrieved = F.normalize(retrieved, dim=-1)
            blended = alpha * centered + (1 - alpha) * retrieved  # (N, D)
            blended = F.normalize(blended, dim=-1)

            W_new = (blended.T @ blended) / blended.shape[0]
            self.W_semantic.mul_(self.semantic_momentum).add_(
                W_new, alpha=1 - self.semantic_momentum
            )
            # Renormalize to maintain constant energy.
            current_norm = self.W_semantic.norm().item()
            if current_norm > 1e-8:
                self.W_semantic.mul_(self._semantic_target_norm / current_norm)

        self._pending_states = None

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape

        # === Episodic retrieval (current sequence) ===
        S_ep = torch.matmul(hidden_states, hidden_states.transpose(-1, -2)) * self.scale

        # Causal mask: position j can only retrieve from i < j
        causal_mask = torch.triu(
            torch.ones(T, T, device=hidden_states.device, dtype=torch.bool), diagonal=0
        )
        S_ep.masked_fill_(causal_mask, float('-inf'))

        # Sparse top-k: keep only the k strongest matches per query
        k_ep = min(self.top_k, T - 1)
        if k_ep > 0:
            topk_vals, topk_idx = S_ep.topk(k_ep, dim=-1)
            S_ep_sparse = torch.zeros_like(S_ep)
            S_ep_sparse.scatter_(-1, topk_idx, topk_vals)
            S_ep_sparse = S_ep_sparse.masked_fill(S_ep_sparse.isinf(), 0.0)
        else:
            S_ep_sparse = torch.zeros_like(S_ep)

        y_episodic = torch.matmul(S_ep_sparse, hidden_states)

        # === Semantic retrieval (accumulated long-term memory) ===
        if self.has_semantic:
            y_semantic = torch.matmul(hidden_states, self.W_semantic) * self.scale
        else:
            y_semantic = torch.zeros_like(hidden_states)

        # === Gated combination ===
        gate = torch.sigmoid((y_episodic * y_semantic).sum(dim=-1, keepdim=True))
        y = gate * y_episodic + (1 - gate) * y_semantic

        return y


class HippoDecoderLayer(nn.Module):
    """
    Decoder layer: Hopfield retrieval modulates the SwiGLU gate of the MLP.
    Memory shapes which cortical circuits fire. Post-MLP residual is stored
    for consolidation, closing the hippocampal-cortical loop.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        top_k = getattr(config, 'hopfield_top_k', 8)
        semantic_momentum = getattr(config, 'semantic_momentum', 0.99)

        self.hopfield = HopfieldLayer(config.hidden_size, top_k, semantic_momentum)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Memory projection: translates retrieval into input modification
        self.mem_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def memory_modulated_mlp(self, x, y_hop):
        """SwiGLU where memory modifies the input before both gate and signal pathways."""
        x_mod = x + self.mem_proj(y_hop)
        gate = F.silu(self.mlp.gate_proj(x_mod))
        up = self.mlp.up_proj(x_mod)
        return self.mlp.down_proj(gate * up)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states

        # Hopfield retrieval (hippocampal read)
        y_hop = self.hopfield(self.input_layernorm(hidden_states))

        # Memory-modulated MLP (cortical computation on memory-blended input)
        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.memory_modulated_mlp(normed, y_hop)

        # Store post-MLP residual for consolidation (hippocampal write of cortical activity)
        if self.training:
            self.hopfield._pending_states = hidden_states.detach()

        return (hidden_states,)


class HippoModel(LlamaPreTrainedModel):
    """
    Hippocampal transformer backbone replacing LlamaModel.
    Stacks HippoDecoderLayer (Hopfield + MLP) instead of LlamaDecoderLayer (Attention + MLP).
    """
    config_class = CALMConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            HippoDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )


class HippoEnergyTransformer(CALM):
    """
    CALM with hippocampal backbone (Hopfield layers) trained with energy score.
    Replaces LlamaModel with HippoModel; everything else (autoencoder,
    generative head, energy loss) stays the same.
    """
    config_class = CALMConfig

    def __init__(self, config):
        super().__init__(config)

        # Frozen autoencoder
        self.ae_config = AutoencoderConfig.from_pretrained(config.ae_path)
        self.ae_model = Autoencoder.from_pretrained(config.ae_path, config=self.ae_config)
        for param in self.ae_model.parameters():
            param.requires_grad = False
        self.ae_model.eval()

        # Hippocampal backbone (replaces LlamaModel)
        self.transformer = HippoModel(config)

        # Generative head (energy-based)
        self.mlp_generator = MLPGenerator(config)
        self.generative_head = self.mlp_generator

        self.padding_idx = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.patch_size = config.patch_size

        # Input compression: K token embeddings -> single patch vector
        self.embed_proj = nn.Sequential(
            nn.Linear(self.patch_size * config.hidden_size, 2 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6),
        )

        self.post_init()
        self.mlp_generator.initialize_weights()
        self.noise_size = config.noise_size
        self.beta = config.beta
        self.num_samples = config.num_samples

    def distance(self, x_1, x_2):
        return torch.pow(torch.linalg.norm(x_1 - x_2, ord=2, dim=-1), self.beta)

    def energy_score(self, x, mean, log_std):
        n_x = x.shape[0]
        x_i = x.unsqueeze(1)
        x_j = x.unsqueeze(0)
        distance_matrix = self.distance(x_i, x_j)
        distance_x = distance_matrix.sum(dim=(0, 1)) / (n_x * (n_x - 1))

        std = torch.exp(log_std)
        n_y = 100
        eps = torch.randn((n_y, *mean.shape), device=mean.device)
        y = mean + eps * std

        x_ = x.reshape(n_x, 1, *x.shape[1:])
        y_ = y.reshape(1, n_y, *y.shape[1:])
        distance_y = self.distance(x_, y_).mean(dim=(0, 1))

        score = distance_x - distance_y * 2
        return score

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        batch_size, seq_length = input_ids.size()
        patch_size = self.patch_size
        latent_length = seq_length // patch_size

        labels = labels[:, patch_size:]
        mask = labels.ne(-100)
        labels = labels[mask].unsqueeze(0)

        # Ground-truth latents from frozen autoencoder
        latent_states = self.ae_model.encoder(input_ids=labels)
        latent_states = latent_states.squeeze(0)
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)

        # Patch embeddings
        inputs_embeds = self.transformer.embed_tokens(input_ids).reshape(
            batch_size, latent_length, -1
        )[:, :-1, :]
        inputs_embeds = self.embed_proj(inputs_embeds)

        # Forward through hippocampal backbone
        outputs = self.transformer(inputs_embeds=inputs_embeds)
        hidden_states = outputs[0]

        patch_mask = mask.reshape(batch_size, latent_length - 1, patch_size)[:, :, 0]
        hidden_states = hidden_states[patch_mask]

        # Generate predictions
        hidden_states_repeated = hidden_states.unsqueeze(0).repeat(self.num_samples, 1, 1)
        latent_predictions = self.generative_head.sample(hidden_states_repeated)

        # Energy loss
        loss = -self.energy_score(latent_predictions, mean, log_std)
        loss = loss.mean()

        # NOTE: call model.consolidate() AFTER loss.backward() + optimizer.step()
        # to deposit this sequence's patterns into semantic memory.

        # TODO: eval_brier requires KV caching which HippoModel doesn't support yet.
        return CustomCausalLMOutput(loss=loss)

    @torch.no_grad()
    def consolidate(self):
        """Deposit episodic patterns from the last forward pass into semantic memory."""
        for layer in self.transformer.layers:
            layer.hopfield.consolidate()
