"""Eve compact temporal transformer — the model that must beat the baselines.

Design, grounded in the literature:

  - **Encoder-only, compact, regularized.** The 2025 consensus for time-series
    classification on small / noisy data (PatchTST and follow-ups; "Automating
    Versatile Time-Series Analysis with Tiny Transformers"): an encoder-only
    transformer is a strong unified backbone, but channel-dependent transformers
    *overfit noise*, so start lightweight and add capacity only if validation
    improves. Hence a small d_model, 1–2 layers, dropout + weight decay, and a
    learned positional embedding over the bar window.
  - **Cross-entropy on cost-aware labels.** CE remains the pragmatic default for
    classification; the target is the cost-aware action label from
    :mod:`src.eve_labels` (down / no-trade / up), not raw direction.
  - **Judged on post-cost expectancy, not accuracy.** The model exposes the same
    ``fit(train, cost)`` / ``predict(test)`` / ``predict_proba(test)`` interface
    as the baselines, so it is dropped straight into the `eve_baselines`
    walk-forward harness and scored on the *same* out-of-sample post-cost series.
    The literature's own warning (TLOB / FI-2010 / LOBCAST) is that LOB/bar
    transformers look strong on accuracy and collapse after costs — so this code
    assumes nothing; the harness reports the truth.

torch is decoupled from numpy here (features are native Python lists → torch
tensors), so it runs under torch 2.2 + numpy 2 without the numpy bridge.
"""

from __future__ import annotations

import warnings
from typing import Sequence

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from .eve_data import EveSequenceSample, select_torch_device  # noqa: E402
from .eve_labels import DOWN, FLAT, UP, cost_aware_label  # noqa: E402

_ACTIONS = (DOWN, FLAT, UP)


class _Encoder(nn.Module):
    """Small encoder-only transformer over a window of bar-feature vectors."""

    def __init__(
        self,
        n_features: int,
        window: int,
        *,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 96,
        dropout: float = 0.2,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, window, F)
        h = self.input(x) + self.pos
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))  # mean-pool over time
        return self.head(self.dropout(h))


class TransformerModel:
    """Baseline-protocol wrapper: ``fit(train, cost)`` / ``predict(test)``.

    Re-initializes weights on every ``fit`` so the walk-forward harness can refit
    per fold with no leakage across folds.
    """

    name = "transformer"

    def __init__(
        self,
        *,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 96,
        dropout: float = 0.2,
        epochs: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_train: int = 80_000,
        val_frac: float = 0.15,
        patience: int = 3,
        balanced: bool = True,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_train = max_train
        self.val_frac = val_frac
        self.patience = patience
        self.balanced = balanced
        self.device = device or select_torch_device(torch)
        self.seed = seed
        self._model: _Encoder | None = None
        self._mu: torch.Tensor | None = None
        self._sd: torch.Tensor | None = None
        self.history: dict = {}

    # -- tensor helpers (no numpy) -----------------------------------------
    def _to_tensor(self, samples: Sequence[EveSequenceSample]) -> torch.Tensor:
        return torch.tensor([s.features for s in samples], dtype=torch.float32)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mu) / self._sd

    # -- protocol ----------------------------------------------------------
    def fit(self, train: Sequence[EveSequenceSample], cost: float) -> "TransformerModel":
        torch.manual_seed(self.seed)
        if not train:
            return self
        ordered = list(train)
        # Subsample very large folds by uniform stride (keeps chronology).
        if len(ordered) > self.max_train:
            stride = len(ordered) / self.max_train
            ordered = [ordered[int(i * stride)] for i in range(self.max_train)]

        X = self._to_tensor(ordered)  # (N, T, F)
        y = torch.tensor(
            [_ACTIONS.index(cost_aware_label(s.future_return, cost)) for s in ordered],
            dtype=torch.long,
        )
        n, window, n_features = X.shape
        flat = X.reshape(-1, n_features)
        self._mu = flat.mean(dim=0)
        sd = flat.std(dim=0)
        self._sd = torch.where(sd > 0, sd, torch.ones_like(sd))
        X = self._standardize(X)

        # Chronological train/val split for early stopping.
        n_val = max(1, int(n * self.val_frac)) if n > 10 else 0
        n_tr = n - n_val
        Xtr, ytr = X[:n_tr], y[:n_tr]
        Xva, yva = (X[n_tr:], y[n_tr:]) if n_val else (X[:0], y[:0])

        dev = torch.device(self.device)
        model = _Encoder(
            n_features, window, d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, dim_ff=self.dim_ff, dropout=self.dropout,
        ).to(dev)
        Xtr, ytr = Xtr.to(dev), ytr.to(dev)
        if n_val:
            Xva, yva = Xva.to(dev), yva.to(dev)

        weight = None
        if self.balanced:
            counts = torch.bincount(ytr, minlength=3).float()
            inv = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))
            weight = (inv / inv.sum() * 3.0).to(dev)  # mean ~1
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = float("inf")
        best_state = None
        bad = 0
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        losses: list[float] = []
        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(n_tr, generator=g)
            ep_loss = 0.0
            for i in range(0, n_tr, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = model(Xtr[idx])
                loss = loss_fn(out, ytr[idx])
                loss.backward()
                opt.step()
                ep_loss += float(loss.item()) * len(idx)
            losses.append(ep_loss / max(1, n_tr))
            # Early stopping on validation loss.
            if n_val:
                model.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(model(Xva), yva).item())
                if vloss < best_val - 1e-5:
                    best_val, bad = vloss, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model.to(torch.device("cpu"))
        self.history = {"epochs_run": len(losses), "train_loss": losses, "best_val": best_val}
        return self

    @torch.no_grad()
    def predict_proba(self, test: Sequence[EveSequenceSample]) -> list[list[float]]:
        if self._model is None or not test:
            return [[1 / 3, 1 / 3, 1 / 3] for _ in test]
        X = self._standardize(self._to_tensor(test))
        logits = self._model(X)
        return torch.softmax(logits, dim=1).tolist()

    def predict(self, test: Sequence[EveSequenceSample]) -> list[int]:
        proba = self.predict_proba(test)
        return [_ACTIONS[max(range(3), key=lambda k: row[k])] for row in proba]


def default_transformer(**kwargs) -> TransformerModel:
    return TransformerModel(**kwargs)
