"""Small dependency-light replacements for the two sklearn pieces we use (roc_auc_score, LogisticRegression),
so the research code runs anywhere torch runs (including machines that block sklearn's compiled extensions)."""
from __future__ import annotations

import numpy as np
import torch


def roc_auc_score(y, s) -> float:
    """Binary AUC via average ranks (ties handled), equals sklearn's roc_auc_score."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        raise ValueError("AUC needs both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


class LogisticRegression:
    """L2-regularized multinomial/binary logistic regression, objective = sum CE + ||W||^2 / (2C) (same as sklearn)."""

    def __init__(self, C: float = 1.0, max_iter: int = 500, device: str = "cpu"):
        self.C, self.max_iter, self.device = C, max_iter, device
        self.dtype = torch.float64 if device == "cpu" else torch.float32   # fp64 is slow on GPUs

    def fit(self, X, y):
        X = torch.as_tensor(np.asarray(X), dtype=self.dtype, device=self.device)
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        yt = torch.as_tensor(yi, device=self.device)
        k = len(self.classes_)
        out = 1 if k == 2 else k
        W = torch.zeros(X.shape[1], out, dtype=self.dtype, device=self.device, requires_grad=True)
        b = torch.zeros(out, dtype=self.dtype, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=self.max_iter, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-6, tolerance_change=1e-10)

        def closure():
            opt.zero_grad()
            z = X @ W + b
            ce = torch.nn.functional.binary_cross_entropy_with_logits(z.squeeze(1), yt.to(self.dtype), reduction="sum") \
                if k == 2 else torch.nn.functional.cross_entropy(z, yt, reduction="sum")
            loss = ce + W.pow(2).sum() / (2 * self.C)
            loss.backward()
            return loss

        opt.step(closure)
        self.W, self.b = W.detach(), b.detach()
        return self

    def predict_proba(self, X):
        z = torch.as_tensor(np.asarray(X), dtype=self.dtype, device=self.device) @ self.W + self.b
        p = torch.sigmoid(z) if len(self.classes_) == 2 else torch.softmax(z, 1)
        p = p.double().cpu().numpy()
        return np.column_stack([1 - p[:, 0], p[:, 0]]) if len(self.classes_) == 2 else p

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def score(self, X, y) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())
