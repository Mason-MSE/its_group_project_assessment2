"""Simple baseline models used in the accompanying report.

* :class:`HistoricalAverage` – predicts every future step using the
  seasonal mean of the corresponding time-of-week slot.  This baseline
  is deterministic and requires no training.
* :class:`LSTMBaseline` – a per-sensor LSTM implemented in PyTorch that
  ignores spatial dependencies.  Included to illustrate the gain
  obtained by moving to graph-based models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except Exception:  # pragma: no cover - torch is optional for baselines
    _TORCH_OK = False


class HistoricalAverage:
    """Seasonal Historical Average baseline.

    The forecast at horizon ``h`` for sensor ``i`` at time ``t`` is the
    mean of the training-set observations at ``(t + h) mod P`` for
    sensor ``i``, where ``P`` is the seasonality period (default: one
    week at 5-minute resolution = ``24 * 12 * 7``).
    """

    def __init__(self, period: int = 24 * 12 * 7) -> None:
        self.period = period
        self._table: Optional[np.ndarray] = None

    def fit(self, speed: np.ndarray) -> "HistoricalAverage":
        """Fit on a ``(T, N)`` speed tensor from the training split."""
        T, N = speed.shape
        table = np.zeros((self.period, N), dtype=np.float32)
        counts = np.zeros((self.period, 1), dtype=np.float32)
        for t in range(T):
            slot = t % self.period
            table[slot] += speed[t]
            counts[slot] += 1.0
        counts[counts == 0] = 1.0
        self._table = table / counts
        return self

    def predict(self, start_step: int, horizon: int, num_windows: int) -> np.ndarray:
        """Return predictions of shape ``(num_windows, horizon, N, 1)``."""
        if self._table is None:
            raise RuntimeError("HistoricalAverage.fit must be called first.")
        out = np.zeros((num_windows, horizon, self._table.shape[1], 1),
                       dtype=np.float32)
        for w in range(num_windows):
            for h in range(horizon):
                slot = (start_step + w + h) % self.period
                out[w, h, :, 0] = self._table[slot]
        return out


if _TORCH_OK:

    class LSTMBaseline(nn.Module):
        """A per-sensor LSTM baseline that ignores spatial structure."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            horizon: int,
            num_layers: int = 2,
        ) -> None:
            super().__init__()
            self.horizon = horizon
            self.output_dim = output_dim
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                    batch_first=True)
            self.head = nn.Linear(hidden_dim, horizon * output_dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Predict ``horizon`` steps for every sensor.

            Parameters
            ----------
            inputs : torch.Tensor
                Shape ``(B, T, N, F)``.

            Returns
            -------
            torch.Tensor
                Shape ``(B, horizon, N, output_dim)``.
            """
            B, T, N, F = inputs.shape
            # Fold nodes into the batch dimension and run the LSTM once.
            reshaped = inputs.permute(0, 2, 1, 3).contiguous().view(B * N, T, F)
            _, (h_n, _) = self.encoder(reshaped)
            last = h_n[-1]  # (B * N, hidden)
            out = self.head(last).view(B, N, self.horizon, self.output_dim)
            return out.permute(0, 2, 1, 3).contiguous()
else:  # pragma: no cover
    class LSTMBaseline:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is not installed; install torch to use LSTMBaseline."
            )
