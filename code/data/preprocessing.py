"""Preprocessing utilities shared by all model back-ends.

The two responsibilities of this module are:

1. **Z-score scaling** – standardise the observed speed channel using
   training-set statistics and provide an inverse transform for reporting.
2. **Adjacency matrix construction** – build a weighted directed graph
   from a pairwise distance matrix using the thresholded Gaussian kernel
   introduced by Li et al. (2018):

   .. math::

       A_{ij} = \\exp(-d_{ij}^{2} / \\sigma^{2}) \\; \\text{if } d_{ij} \\le \\kappa
       \\; \\text{else } 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class StandardScaler:
    """Simple Z-score scaler for the speed channel."""

    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + 1e-6) + self.mean


def fit_scaler(train_features: np.ndarray) -> StandardScaler:
    """Compute mean/std of the *speed* channel (last dim index 0)."""
    speed = train_features[..., 0]
    return StandardScaler(mean=float(np.nanmean(speed)),
                          std=float(np.nanstd(speed)))


def build_adjacency_matrix(
    distance: np.ndarray,
    sigma: float | None = None,
    threshold: float = 0.1,
) -> np.ndarray:
    """Return a thresholded Gaussian-kernel adjacency matrix.

    Parameters
    ----------
    distance : np.ndarray
        Pairwise distances of shape ``(N, N)``.
    sigma : float, optional
        Kernel width.  Defaults to the standard deviation of positive
        entries in ``distance``.
    threshold : float
        Values below this after applying the kernel are zeroed out to
        enforce sparsity (default 0.1, following Li et al., 2018).
    """
    if sigma is None:
        positive = distance[distance > 0]
        sigma = float(positive.std()) if positive.size else 1.0
    weights = np.exp(-(distance ** 2) / (sigma ** 2))
    weights[weights < threshold] = 0.0
    np.fill_diagonal(weights, 0.0)
    return weights.astype(np.float32)


def dual_random_walk_matrices(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the forward and backward random-walk transition matrices.

    The DCRNN diffusion convolution needs ``D_O^{-1} A`` (forward) and
    ``D_I^{-1} A^T`` (backward).
    """
    out_deg = adj.sum(axis=1, keepdims=True)
    out_deg[out_deg == 0] = 1.0
    forward = adj / out_deg

    in_deg = adj.sum(axis=0, keepdims=True)
    in_deg[in_deg == 0] = 1.0
    backward = adj.T / in_deg.T
    return forward.astype(np.float32), backward.astype(np.float32)
