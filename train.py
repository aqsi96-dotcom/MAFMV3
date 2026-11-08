"""Train MAFMv3 for spine lesion classification (VinDr-SpineXR)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models import MAFMv3
from utils import build_dataloaders, create_dummy_data, evaluate_classifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train MAFMv3")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dummy-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dummy_data:
        create_dummy_data("data/vindr_spinexr")
        cfg["data"]["train_dir"] = "data/vindr_spinexr/train"
        cfg["data"]["val_dir"] = "data/vindr_spinexr/val"
        cfg["data"]["num_workers"] = 0
        cfg["train"]["batch_size"] = min(cfg["train"]["batch_size"], 2)
        cfg["model"]["pretrained"] = False

    train_loader, val_loader = build_dataloaders(cfg)
    if len(train_loader.dataset) == 0:
        raise RuntimeError("No training images found. Expected class folders under train_dir.")

    model = MAFMv3(
        num_classes=cfg["data"]["num_classes"],
        in_channels=3,
        pretrained=cfg["model"]["pretrained"],
        aspp_rates=cfg["model"]["aspp_rates"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    patience = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
            x1, x2, x3 = batch["x1"].to(device), batch["x2"].to(device), batch["x3"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x1, x2, x3)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)
        val_metrics = evaluate_classifier(model, val_loader, device, cfg["data"]["num_classes"])
        scheduler.step(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_acc={val_metrics['accuracy']:.2f}% prec={val_metrics['precision']:.2f}% "
            f"rec={val_metrics['recall']:.2f}% f1={val_metrics['f1']:.2f}% "
            f"auc={val_metrics['auc']:.2f}% fnr={val_metrics['fnr']:.2f}%"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "metrics": val_metrics,
                    "config": cfg,
                },
                ckpt_dir / "mafmv3_best.pt",
            )
            print(f"  saved mafmv3_best.pt (acc={best_acc:.2f}%)")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()
