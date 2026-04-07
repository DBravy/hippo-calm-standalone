#!/usr/bin/env python3
"""
Train CALM with LCA backbone on wikitext.

Two conditions run sequentially:
  1. Baseline: standard attention -> MLP
  2. LCA: attention -> LCA(replace) -> MLP

Same config, same data, same training schedule.
Compare energy loss curves.

Usage:
  python test_lca_calm.py
  python test_lca_calm.py --save my_run_name
  python test_lca_calm.py --condition lca       # run only LCA
  python test_lca_calm.py --condition baseline   # run only baseline
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from transformers import AutoTokenizer
from models.configuration_calm import CALMConfig
from models.modeling_lca_calm import LCAEnergyTransformer


def load_wikitext(path, tokenizer, block_size):
    """Load wikitext JSON, tokenize, chunk into fixed-length sequences."""
    with open(path) as f:
        docs = [json.loads(line)["text"] for line in f]

    all_ids = []
    for doc in docs:
        ids = tokenizer.encode(doc, add_special_tokens=False)
        all_ids.extend(ids)

    num_chunks = len(all_ids) // block_size
    chunks = [all_ids[i * block_size : (i + 1) * block_size]
              for i in range(num_chunks)]

    print(f"Tokenized: {len(all_ids):,} tokens -> {num_chunks} chunks of {block_size}")
    return torch.tensor(chunks, dtype=torch.long)


def train_condition(config, train_data, val_data, use_lca, lca_config,
                    device, batch_size, num_epochs, label):
    """Train one condition and return metrics."""

    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"{'='*60}")

    model = LCAEnergyTransformer(config, use_lca=use_lca, lca_config=lca_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}  (total: {total:,})")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )

    steps_per_epoch = (train_data.shape[0] - batch_size + 1) // batch_size
    total_steps = steps_per_epoch * num_epochs
    log_interval = max(1, total_steps // 500)

    print(f"Training: {num_epochs} epochs, batch_size={batch_size}, "
          f"{steps_per_epoch} steps/epoch, {total_steps} total")
    print(f"{'epoch':>5}  {'step':>6}  {'loss':>10}  {'grad':>10}  {'ms':>6}")
    print("-" * 50)

    run_data = {
        "label": label,
        "use_lca": use_lca,
        "lca_config": lca_config,
        "trainable_params": trainable,
        "steps": [],
        "epochs": [],
    }

    global_step = 0
    for epoch in range(num_epochs):
        perm = torch.randperm(train_data.shape[0])
        train_shuffled = train_data[perm]

        epoch_losses = []
        for i in range(0, train_shuffled.shape[0] - batch_size + 1, batch_size):
            batch = train_shuffled[i : i + batch_size].to(device)
            labels = batch.clone()

            t0 = time.time()
            optimizer.zero_grad()
            outputs = model(input_ids=batch, labels=labels)
            loss = outputs.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            dt = (time.time() - t0) * 1000

            step_loss = loss.item()
            step_grad = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            epoch_losses.append(step_loss)

            if global_step % log_interval == 0:
                run_data["steps"].append({
                    "epoch": epoch, "step": global_step,
                    "loss": step_loss, "grad_norm": step_grad, "ms": dt,
                })
                print(f"{epoch:5d}  {global_step:6d}  {step_loss:10.4f}  "
                      f"{step_grad:10.4f}  {dt:6.0f}")

            global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, val_data.shape[0] - batch_size + 1, batch_size):
                batch = val_data[i : i + batch_size].to(device)
                outputs = model(input_ids=batch, labels=batch.clone())
                val_losses.append(outputs.loss.item())
        val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')
        model.train()

        print(f"  --> epoch {epoch}: train={avg_loss:.4f}, val={val_loss:.4f}")

        run_data["epochs"].append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_loss,
        })

    print(f"\n  [{label}] Final: train={avg_loss:.4f}, val={val_loss:.4f}")
    run_data["final_train_loss"] = avg_loss
    run_data["final_val_loss"] = val_loss

    return run_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", nargs="?", const="", default=None,
                        help="Save with optional name")
    parser.add_argument("--condition", type=str, default="both",
                        choices=["baseline", "lca", "both"],
                        help="Which condition(s) to run")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lca_steps", type=int, default=10,
                        help="LCA iteration steps")
    parser.add_argument("--lca_lambda", type=float, default=0.1,
                        help="LCA sparsity threshold")
    parser.add_argument("--lca_mult", type=int, default=2,
                        help="LCA overcomplete multiplier (d_lca = hidden * mult)")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name()})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Data
    block_size = 512
    tokenizer = AutoTokenizer.from_pretrained("./llama3_tokenizer")
    data = load_wikitext(
        "./data/wikitext_document_level-test.json", tokenizer, block_size
    )
    perm = torch.randperm(data.shape[0])
    data = data[perm]
    split = max(1, int(data.shape[0] * 0.9))
    train_data = data[:split]
    val_data = data[split:]
    print(f"Train: {train_data.shape[0]} chunks, Val: {val_data.shape[0]} chunks")

    # Config (same for both conditions)
    config = CALMConfig(
        ae_path="./autoencoder",
        model_type="lca_calm",
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
        noise_size=64,
        num_mlp_layers=4,
        num_samples=4,
        beta=1.0,
    )

    lca_config = {
        "d_lca": config.hidden_size * args.lca_mult,
        "lca_steps": args.lca_steps,
        "lca_lambda": args.lca_lambda,
    }

    print(f"\nBackbone: hidden={config.hidden_size}, layers={config.num_hidden_layers}")
    print(f"LCA: d_lca={lca_config['d_lca']}, steps={lca_config['lca_steps']}, "
          f"lambda={lca_config['lca_lambda']}")

    # Run conditions
    all_results = {}

    if args.condition in ("baseline", "both"):
        all_results["baseline"] = train_condition(
            config, train_data, val_data,
            use_lca=False, lca_config=None,
            device=device, batch_size=args.batch_size,
            num_epochs=args.epochs, label="Baseline",
        )

    if args.condition in ("lca", "both"):
        all_results["lca"] = train_condition(
            config, train_data, val_data,
            use_lca=True, lca_config=lca_config,
            device=device, batch_size=args.batch_size,
            num_epochs=args.epochs, label="LCA Replace",
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"  {name:15s}: params={res['trainable_params']:,}, "
              f"final_val={res['final_val_loss']:.4f}")

    # Save
    os.makedirs("runs", exist_ok=True)
    if args.save is not None:
        name = args.save or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"runs/{name}"
        os.makedirs(run_dir, exist_ok=True)
        log_path = f"{run_dir}/log.json"
    else:
        log_path = "runs/lca_calm_latest.json"

    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Log saved to {log_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
