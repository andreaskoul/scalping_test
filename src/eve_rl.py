"""Direct-RL portfolio allocator (Differential Sharpe Ratio objective).

The RL half of Eve's ensemble. A small, asset-shared policy network maps each
asset's cross-sectional signal vector plus its current holding to a score; scores
become a dollar-neutral, leverage-capped long-short book. The policy is trained by
**direct / recurrent reinforcement learning** (Moody & Saffell, NIPS 1998):
roll the book forward over the training window, accumulate the per-step
**Differential Sharpe Ratio** of the *post-cost* portfolio return, and ascend its
gradient. Because previous holdings feed back in, the policy learns to control
turnover (cost). EMA state (A, B) and drifted prior weights are detached each
step, giving the standard stable online (truncated) DSR update — no value-function
instability.

torch is imported lazily and decoupled from numpy (tensors built from Python
floats), so this runs on torch 2.2 + numpy 2.
"""

from __future__ import annotations

import warnings

import numpy as np

from .eve_portfolio import Panel

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")


class DirectRLAllocator:
    """Recurrent-RL long-short allocator trained on cumulative post-cost DSR."""

    name = "rl_allocator"

    def __init__(
        self,
        *,
        hidden: int = 16,
        leverage: float = 1.0,
        eta: float = 0.02,
        epochs: int = 40,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        cost: float = 0.0005,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.leverage = leverage
        self.eta = eta
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.cost = cost
        self.device = device
        self.seed = seed
        self._net = None
        self.history: dict = {}

    def _build(self, n_features: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(n_features + 1, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1),
        )

    def _weights(self, scores):
        # Dollar-neutral, gross leverage = self.leverage.
        import torch

        s = scores - scores.mean()
        denom = torch.clamp(s.abs().sum(), min=1e-6)
        return self.leverage * s / denom

    def fit(self, panel: Panel) -> "DirectRLAllocator":
        import torch

        torch.manual_seed(self.seed)
        dev = torch.device(self.device)
        T, N, F = panel.signals.shape
        if T < 3:
            return self
        S = torch.tensor(panel.signals.tolist(), dtype=torch.float32, device=dev)
        R = torch.tensor(panel.fwd_returns.tolist(), dtype=torch.float32, device=dev)
        net = self._build(F).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup = min(max(2, int(1.0 / self.eta)) if self.eta > 0 else 2, max(2, T // 4))

        losses = []
        for _ in range(self.epochs):
            opt.zero_grad()
            w_prev_eff = torch.zeros(N, device=dev)
            A = B = None
            total = torch.zeros((), device=dev)
            for t in range(T):
                inp = torch.cat([S[t], w_prev_eff.unsqueeze(-1)], dim=-1)  # (N, F+1)
                scores = net(inp).squeeze(-1)
                w = self._weights(scores)
                gross = (w * R[t]).sum()
                turnover = (w - w_prev_eff).abs().sum()
                net_ret = gross - self.cost * turnover
                if A is None:
                    A = net_ret.detach()
                    B = (net_ret ** 2).detach()
                else:
                    dA = net_ret - A
                    dB = net_ret ** 2 - B
                    var = torch.clamp(B - A * A, min=1e-8)
                    if t >= warmup:  # skip unstable EMA warmup
                        total = total + (B * dA - 0.5 * A * dB) / var.pow(1.5)
                    A = (A + self.eta * dA).detach()
                    B = (B + self.eta * dB).detach()
                drift = (1.0 + gross).detach()
                w_prev_eff = ((w * (1.0 + R[t])) / drift).detach()
            loss = -total / T
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        net.eval()
        self._net = net
        self.history = {"epochs": len(losses), "final_neg_dsr": losses[-1] if losses else None}
        return self

    def predict_weights(self, panel: Panel) -> np.ndarray:
        import torch

        T, N, F = panel.signals.shape
        if self._net is None or T == 0:
            return np.zeros((T, N))
        dev = torch.device(self.device)
        S = torch.tensor(panel.signals.tolist(), dtype=torch.float32, device=dev)
        R = torch.tensor(panel.fwd_returns.tolist(), dtype=torch.float32, device=dev)
        W = np.zeros((T, N))
        with torch.no_grad():
            w_prev_eff = torch.zeros(N, device=dev)
            for t in range(T):
                inp = torch.cat([S[t], w_prev_eff.unsqueeze(-1)], dim=-1)
                w = self._weights(self._net(inp).squeeze(-1))
                W[t] = w.cpu().tolist()  # avoid torch<->numpy bridge
                gross = (w * R[t]).sum()
                drift = 1.0 + gross
                w_prev_eff = (w * (1.0 + R[t])) / drift if float(drift) > 1e-9 else w
        return W
