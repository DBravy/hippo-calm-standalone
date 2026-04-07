#!/usr/bin/env python3
"""Train HippoEnergyTransformer on wikitext. Tiny config, MPS/CPU."""

import sys
import os
import json
import time
import argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer
from models.configuration_calm import CALMConfig
from models.modeling_hopfield import HippoEnergyTransformer


def load_wikitext(path, tokenizer, block_size):
    """Load wikitext JSON, tokenize, chunk into fixed-length sequences."""
    with open(path) as f:
        docs = [json.loads(line)["text"] for line in f]

    all_ids = []
    for doc in docs:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        all_ids.extend(ids)

    num_chunks = len(all_ids) // block_size
    chunks = [all_ids[i * block_size : (i + 1) * block_size] for i in range(num_chunks)]

    print(f"Tokenized: {len(all_ids):,} tokens -> {num_chunks} chunks of {block_size}")
    return torch.tensor(chunks, dtype=torch.long)


def get_semantic_diagnostics(model):
    """Compute W_semantic SVD spectrum and effective rank per layer."""
    diagnostics = []
    print("\n  Semantic memory per layer:")
    for i, layer in enumerate(model.transformer.layers):
        h = layer.hopfield
        if not h.has_semantic.item():
            print(f"    layer {i}: inactive")
            diagnostics.append({"layer": i, "active": False})
            continue
        S = torch.linalg.svdvals(h.W_semantic.cpu())
        S_norm = S / S.sum()
        eff_rank = torch.exp(-torch.sum(S_norm * torch.log(S_norm + 1e-10))).item()
        top5 = S[:5].tolist()
        print(f"    layer {i}: ||W||={S.sum():.2f}, eff_rank={eff_rank:.1f}, top5_sv={[f'{v:.3f}' for v in top5]}")
        diagnostics.append({
            "layer": i, "active": True,
            "W_norm": S.sum().item(), "eff_rank": eff_rank,
            "top5_sv": top5, "full_sv": S.tolist(),
        })
    return diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", nargs="?", const="", default=None,
                        help="Save run with optional name (uses timestamp if blank)")
    args = parser.parse_args()

    # --- Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # --- Data ---
    block_size = 512
    tokenizer = AutoTokenizer.from_pretrained("./llama3_tokenizer")
    data = load_wikitext("./data/wikitext_document_level-test.json", tokenizer, block_size)
    perm = torch.randperm(data.shape[0])
    data = data[perm]
    split = max(1, int(data.shape[0] * 0.9))
    train_data = data[:split]
    val_data = data[split:]
    print(f"Train: {train_data.shape[0]} chunks, Val: {val_data.shape[0]} chunks")

    # --- Config ---
    config = CALMConfig(
        ae_path="./autoencoder",
        model_type="hippo",
        vocab_size=128257,
        pad_token_id=128256,
        bos_token_id=128000,
        eos_token_id=128001,
        latent_size=128,
        patch_size=4,
        hidden_size=256,
        intermediate_size=640,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        hopfield_top_k=8,
        semantic_momentum=0.99,
        noise_size=64,
        num_mlp_layers=4,
        num_samples=4,
        beta=1.0,
    )

    print(f"\nBackbone: hidden={config.hidden_size}, layers={config.num_hidden_layers}")
    print(f"Hopfield: top_k={config.hopfield_top_k}, semantic_momentum={config.semantic_momentum}")

    # --- Model ---
    print("\nInstantiating model...")
    model = HippoEnergyTransformer(config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    model = model.to(device)
    model.train()

    # --- Optimizer ---
    batch_size = 4
    num_epochs = 50
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )

    # --- Training ---
    steps_per_epoch = (train_data.shape[0] - batch_size + 1) // batch_size
    total_steps = steps_per_epoch * num_epochs
    # Target ~500 logged steps regardless of run length; log every step for short runs
    log_interval = max(1, total_steps // 500)

    print(f"\nTraining: {num_epochs} epochs, batch_size={batch_size}, "
          f"{steps_per_epoch} steps/epoch, {total_steps} total, log every {log_interval}")
    print(f"{'epoch':>5}  {'step':>5}  {'loss':>10}  {'grad_norm':>10}  {'ms':>6}")
    print("-" * 50)

    run_log = {
        "config": {k: v for k, v in config.to_dict().items()},
        "training": {
            "batch_size": batch_size, "num_epochs": num_epochs, "lr": 3e-4,
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "log_interval": log_interval,
        },
        "trainable_params": trainable,
        "device": str(device),
        "steps": [],
        "epochs": [],
    }

    global_step = 0
    for epoch in range(num_epochs):
        perm = torch.randperm(train_data.shape[0])
        train_data_shuffled = train_data[perm]

        epoch_losses = []
        for i in range(0, train_data_shuffled.shape[0] - batch_size + 1, batch_size):
            batch = train_data_shuffled[i : i + batch_size].to(device)
            labels = batch.clone()

            t0 = time.time()
            optimizer.zero_grad()
            outputs = model(input_ids=batch, labels=labels)
            loss = outputs.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            model.consolidate()
            dt = (time.time() - t0) * 1000

            step_loss = loss.item()
            step_grad = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            epoch_losses.append(step_loss)
            if global_step % log_interval == 0:
                run_log["steps"].append({
                    "epoch": epoch, "step": global_step,
                    "loss": step_loss, "grad_norm": step_grad, "ms": dt,
                })
            print(f"{epoch:5d}  {global_step:5d}  {step_loss:10.4f}  {step_grad:10.4f}  {dt:6.0f}")
            global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, val_data.shape[0] - batch_size + 1, batch_size):
                batch = val_data[i : i + batch_size].to(device)
                outputs = model(input_ids=batch, labels=batch.clone())
                val_losses.append(outputs.loss.item())
        val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')
        model.train()

        print(f"  --> epoch {epoch} avg train={avg_loss:.4f}, val={val_loss:.4f}")
        diagnostics = get_semantic_diagnostics(model)

        run_log["epochs"].append({
            "epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss,
            "semantic_diagnostics": diagnostics,
        })

    print("\nDone.")

    # --- Save run log + weights ---
    os.makedirs("runs", exist_ok=True)
    if args.save is not None:
        name = args.save or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"runs/{name}"
        os.makedirs(run_dir, exist_ok=True)
        log_path = f"{run_dir}/log.json"
        weights_path = f"{run_dir}/weights.pt"
        torch.save(model.state_dict(), weights_path)
        print(f"Weights saved to {weights_path}")
    else:
        log_path = "runs/latest.json"
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"Run log saved to {log_path}")


if __name__ == "__main__":
    main()
