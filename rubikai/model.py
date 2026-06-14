"""
Réseau de neurones qui estime le "coût pour résoudre" (cost-to-go) d'un état.

Architecture : MLP avec un corps partagé, puis deux têtes (comme DeepCube) :
  - tête VALEUR  : v(s) ≈ nombre de coups estimé pour résoudre s
  - tête POLITIQUE : p(s) ≈ quel coup rapproche le plus de la solution

C'est la tête VALEUR qui guide la recherche A*. La politique aide à l'ordonner.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class CubeNet(nn.Module):
    def __init__(self, in_dim: int, n_moves: int, hidden=(512, 512, 256)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.value_head = nn.Linear(d, 1)
        self.policy_head = nn.Linear(d, n_moves)

    def forward(self, x):
        z = self.body(x)
        v = self.value_head(z).squeeze(-1)       # (B,)
        p = self.policy_head(z)                  # (B, n_moves)
        return v, p

    @torch.no_grad()
    def value(self, x):
        self.eval()
        z = self.body(x)
        return self.value_head(z).squeeze(-1)


def encode_batch(colors: np.ndarray, device="cpu") -> torch.Tensor:
    """Encode un lot d'états (B, n_stickers) en one-hot (B, n_stickers*6)."""
    if colors.ndim == 1:
        colors = colors[None, :]
    B, S = colors.shape
    oh = np.zeros((B, S, 6), dtype=np.float32)
    rows = np.arange(B)[:, None]
    cols = np.arange(S)[None, :]
    oh[rows, cols, colors] = 1.0
    return torch.from_numpy(oh.reshape(B, S * 6)).to(device)
