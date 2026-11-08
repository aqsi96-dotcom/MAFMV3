"""Inference with a trained MAFMv3 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from models import MAFMv3
from utils.dataset import histogram_equalization, to_rgb_stack


def prepare_streams(path: str, size: int, use_he: bool = True):
    gray = np.array(Image.open(path).convert("L"), dtype=np.float32)
    gray = cv2.resize(gray, (size, size))
    if gray.max() > 1:
        gray /= 255.0
    normalized = (gray - gray.mean()) / (gray.std() + 1e-6)
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-6)
    he = histogram_equalization(gray) if use_he else gray
    x1 = torch.from_numpy(to_rgb_stack(gray)).unsqueeze(0)
    x2 = torch.from_numpy(to_rgb_stack(normalized.astype(np.float32))).unsqueeze(0)
    x3 = torch.from_numpy(to_rgb_stack(he)).unsqueeze(0)
    return x1, x2, x3


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = MAFMv3(
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,
        aspp_rates=cfg["model"]["aspp_rates"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x1, x2, x3 = prepare_streams(
        args.image,
        cfg["data"]["image_size"],
        cfg["data"]["use_histogram_equalization"],
    )
    logits = model(x1.to(device), x2.to(device), x3.to(device))
    prob = torch.softmax(logits, dim=1)[0]
    idx = int(prob.argmax())
    names = cfg["data"]["class_names"]
    print(f"Prediction: {names[idx]} ({prob[idx] * 100:.2f}%)")
    for i, name in enumerate(names):
        print(f"  {name}: {prob[i] * 100:.2f}%")


if __name__ == "__main__":
    main()
