from __future__ import annotations

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@torch.no_grad()
def evaluate_classifier(model, loader, device, num_classes: int) -> dict:
    model.eval()
    ys, ps, probs = [], [], []
    for batch in loader:
        x1 = batch["x1"].to(device)
        x2 = batch["x2"].to(device)
        x3 = batch["x3"].to(device)
        labels = batch["label"].to(device)
        logits = model(x1, x2, x3)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        ys.extend(labels.cpu().tolist())
        ps.extend(pred.cpu().tolist())
        probs.extend(prob.cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(ys, ps) * 100,
        "precision": precision_score(ys, ps, average="macro", zero_division=0) * 100,
        "recall": recall_score(ys, ps, average="macro", zero_division=0) * 100,
        "f1": f1_score(ys, ps, average="macro", zero_division=0) * 100,
        "fnr": 100.0 - recall_score(ys, ps, average="macro", zero_division=0) * 100,
    }

    try:
        import numpy as np

        y_true = np.array(ys)
        y_prob = np.array(probs)
        if num_classes == 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1]) * 100
        else:
            metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") * 100
    except Exception:
        metrics["auc"] = 0.0

    return metrics
