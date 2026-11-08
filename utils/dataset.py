"""VinDr-SpineXR style dataset with histogram equalization and multi-input streams."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".bmp", ".dcm", ".npy"}


def _list_images(folder: Path):
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    g = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    eq = cv2.equalizeHist(g).astype(np.float32) / 255.0
    return eq


def to_rgb_stack(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=0)


class SpineXRDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 224,
        class_names: list[str] | None = None,
        use_he: bool = True,
        multi_input_mode: str = "preprocess",
        augment: bool = False,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.use_he = use_he
        self.multi_input_mode = multi_input_mode
        self.augment = augment
        self.class_names = class_names or ["normal", "abnormal"]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples = []
        for name in self.class_names:
            folder = self.root / name
            if folder.exists():
                for img in _list_images(folder):
                    self.samples.append((img, self.class_to_idx[name]))

    def __len__(self):
        return len(self.samples)

    def _load_gray(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32)
        else:
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
        arr = cv2.resize(arr, (self.image_size, self.image_size))
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    def _augment(self, gray: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            gray = np.flip(gray, axis=1).copy()
        if np.random.rand() < 0.5:
            gray = np.flip(gray, axis=0).copy()
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-15, 15)
            m = cv2.getRotationMatrix2D((self.image_size / 2, self.image_size / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, m, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
        if np.random.rand() < 0.3:
            gray = np.clip(gray * np.random.uniform(0.85, 1.15), 0, 1)
        return gray

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        gray = self._load_gray(path)
        if self.augment:
            gray = self._augment(gray)

        if self.multi_input_mode == "views":
            # Expect optional sibling folders for multi-plane inputs
            stem = path.name
            coronal = path.parent.parent / "coronal" / path.parent.name / stem
            sagittal = path.parent.parent / "sagittal" / path.parent.name / stem
            g2 = self._load_gray(coronal) if coronal.exists() else gray
            g3 = self._load_gray(sagittal) if sagittal.exists() else gray
            x1 = to_rgb_stack(gray)
            x2 = to_rgb_stack(g2)
            x3 = to_rgb_stack(g3)
        else:
            # Abstract setting: raw, normalized, histogram-equalized
            raw = gray
            normalized = (gray - gray.mean()) / (gray.std() + 1e-6)
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-6)
            he = histogram_equalization(gray) if self.use_he else gray
            x1 = to_rgb_stack(raw)
            x2 = to_rgb_stack(normalized.astype(np.float32))
            x3 = to_rgb_stack(he)

        return {
            "x1": torch.from_numpy(x1.astype(np.float32)),
            "x2": torch.from_numpy(x2.astype(np.float32)),
            "x3": torch.from_numpy(x3.astype(np.float32)),
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }


def build_dataloaders(cfg: dict):
    train_ds = SpineXRDataset(
        cfg["data"]["train_dir"],
        image_size=cfg["data"]["image_size"],
        class_names=cfg["data"]["class_names"],
        use_he=cfg["data"]["use_histogram_equalization"],
        multi_input_mode=cfg["data"]["multi_input_mode"],
        augment=True,
    )
    val_ds = SpineXRDataset(
        cfg["data"]["val_dir"],
        image_size=cfg["data"]["image_size"],
        class_names=cfg["data"]["class_names"],
        use_he=cfg["data"]["use_histogram_equalization"],
        multi_input_mode=cfg["data"]["multi_input_mode"],
        augment=False,
    )
    nw = cfg["data"]["num_workers"]
    bs = cfg["train"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
    )


def create_dummy_data(root: str = "data/vindr_spinexr", n_train: int = 8, n_val: int = 4):
    root = Path(root)
    for split, n in (("train", n_train), ("val", n_val)):
        for cls in ("normal", "abnormal"):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
                Image.fromarray(img).save(d / f"{cls}_{i:03d}.png")
