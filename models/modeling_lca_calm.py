"""
modeling_lca_calm.py

Standard Llama attention + LCA(replace) + MLP decoder layer,
plugged into the CALM energy-based training framework.

Drop this into your models/ directory alongside modeling_hopfield.py.

Two conditions via use_lca flag:
  - use_lca=False: standard attention -> MLP (baseline)
  - use_lca=True:  attention -> LCA(replace) -> MLP
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaRotaryEmbedding,
)

from .configuration_calm import CALMConfig
from .configuration_autoencoder import AutoencoderConfig
from .modeling_autoencoder import Autoencoder
from .modeling_calm import CALM, CustomCausalLMOutput
from .modeling_energy import MLPGenerator


# ============================================================
# LCA Layer
# ============================================================

class LCALayer(nn.Module):
    """
    Locally Competitive Algorithm layer.

    Projects input to an overcomplete space (d_model -> d_lca),
    runs iterative lateral competition, projects back (d_lca -> d_model).

    When used as a 'replace' operation, the output replaces the residual
    stream, forcing downstream computation to operate on the
    competition-resolved representation.
    """

    def __init__(self, d_model, d_lca, n_steps=10, dt=0.01, tau=0.1, lam=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_model, d_lca) * 0.02)
        self.n_steps = n_steps
        self.dt = dt
        self.tau = tau
        self.lam = lam

    def forward(self, x):
        orig_shape = x.shape
        D = orig_shape[-1]
        x_flat = x.reshape(-1, D)

        # Feedforward drive
        b = x_flat @ self.W

        # Lateral connectivity (weight overlap, zero self-connections)
        G = self.W.T @ self.W
        G = G - torch.diag(torch.diag(G))

        # Iterate dynamics
        u = torch.zeros(x_flat.size(0), self.W.size(1),
                         device=x.device, dtype=x.dtype)
        for _ in range(self.n_steps):
            a = torch.relu(u - self.lam)
            du = (1.0 / self.tau) * (b - a @ G - u)
            u = u + self.dt * du

        a = torch.relu(u - self.lam)

        # Project back to model dimension
        out = a @ self.W.T
        return out.reshape(orig_shape)


# ============================================================
# Decoder layers
# ============================================================

class LCADecoderLayer(nn.Module):
    """
    Attention -> LCA(replace) -> MLP

    Standard causal self-attention, then LCA replaces the residual
    stream with a competition-resolved representation, then MLP
    computes over the clean signal.
    """

    def __init__(self, config, layer_idx, lca_config=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Standard Llama attention
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # LCA between attention and MLP
        lca_config = lca_config or {}
        d_lca = lca_config.get('d_lca', config.hidden_size * 2)
        lca_steps = lca_config.get('lca_steps', 10)
        lca_lambda = lca_config.get('lca_lambda', 0.1)
        self.lca = LCALayer(
            config.hidden_size, d_lca,
            n_steps=lca_steps, lam=lca_lambda,
        )

        # Standard Llama MLP
        self.mlp = LlamaMLP(config)

        # Layer norms
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lca_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states

        # Attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(attn_output, tuple):
            hidden_states = attn_output[0]
        else:
            hidden_states = attn_output

        hidden_states = residual + hidden_states

        # LCA replace: competition resolves the post-attention representation
        hidden_states = self.lca(self.lca_layernorm(hidden_states))

        # MLP on the clean, competition-resolved representation
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class BaselineDecoderLayer(nn.Module):
    """
    Standard attention -> MLP (no LCA). Control condition.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(attn_output, tuple):
            hidden_states = attn_output[0]
        else:
            hidden_states = attn_output
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


# ============================================================
# Backbone model
# ============================================================

class LCAModel(LlamaPreTrainedModel):
    """
    Transformer backbone with optional LCA layers.
    Stacks either LCADecoderLayer or BaselineDecoderLayer.
    """
    config_class = CALMConfig

    def __init__(self, config, use_lca=True, lca_config=None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_lca = use_lca

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        if use_lca:
            self.layers = nn.ModuleList([
                LCADecoderLayer(config, layer_idx=i, lca_config=lca_config)
                for i in range(config.num_hidden_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                BaselineDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ])

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

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
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        batch_size, seq_len, _ = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=hidden_states.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Compute rotary embeddings once
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        for layer in self.layers:
            layer_out = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0]

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )


# ============================================================
# Full CALM model with LCA backbone
# ============================================================

class LCAEnergyTransformer(CALM):
    """
    CALM with standard attention + optional LCA, trained with energy score.

    use_lca=True:  attention -> LCA(replace) -> MLP
    use_lca=False: attention -> MLP (baseline)
    """
    config_class = CALMConfig

    def __init__(self, config, use_lca=True, lca_config=None):
        super().__init__(config)

        # Frozen autoencoder
        self.ae_config = AutoencoderConfig.from_pretrained(config.ae_path)
        self.ae_model = Autoencoder.from_pretrained(
            config.ae_path, config=self.ae_config
        )
        for param in self.ae_model.parameters():
            param.requires_grad = False
        self.ae_model.eval()

        # Backbone
        self.transformer = LCAModel(config, use_lca=use_lca, lca_config=lca_config)

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
    ) -> Union[Tuple, CustomCausalLMOutput]:

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

        # Forward through backbone
        outputs = self.transformer(inputs_embeds=inputs_embeds)
        hidden_states = outputs[0]

        patch_mask = mask.reshape(batch_size, latent_length - 1, patch_size)[:, :, 0]
        hidden_states = hidden_states[patch_mask]

        # Generate predictions
        hidden_states_repeated = hidden_states.unsqueeze(0).repeat(
            self.num_samples, 1, 1
        )
        latent_predictions = self.generative_head.sample(hidden_states_repeated)

        # Energy loss
        loss = -self.energy_score(latent_predictions, mean, log_std)
        loss = loss.mean()

        return CustomCausalLMOutput(loss=loss)
