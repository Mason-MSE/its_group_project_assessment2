"""Encoder–decoder DCRNN (Li et al., 2018).

The encoder consumes a historical window of shape ``(B, T_in, N, F)``
and produces a set of hidden states.  The decoder unrolls
autoregressively for ``horizon`` steps and emits a scalar speed
prediction per sensor and per step.  Scheduled sampling is applied
during training to bridge the gap between teacher forcing and
autoregressive inference.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.dcrnn_cell import DCGRUCell


class DCRNNEncoder(nn.Module):
    def __init__(
        self,
        supports: Sequence[torch.Tensor],
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_diffusion_step: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.cells.append(
                DCGRUCell(supports, in_dim, hidden_dim, max_diffusion_step)
            )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Return the last hidden state of every layer.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(B, T, N, F)``.
        """
        batch, seq_len, num_nodes, _ = inputs.shape
        device = inputs.device
        hidden_states = [cell.init_hidden(batch, num_nodes, device) for cell in self.cells]

        for t in range(seq_len):
            layer_input = inputs[:, t]
            for i, cell in enumerate(self.cells):
                hidden_states[i] = cell(layer_input, hidden_states[i])
                layer_input = hidden_states[i]
        return hidden_states


class DCRNNDecoder(nn.Module):
    def __init__(
        self,
        supports: Sequence[torch.Tensor],
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_diffusion_step: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = output_dim if layer == 0 else hidden_dim
            self.cells.append(
                DCGRUCell(supports, in_dim, hidden_dim, max_diffusion_step)
            )
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        encoder_state: list[torch.Tensor],
        horizon: int,
        teacher: torch.Tensor | None = None,
        teacher_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive decoding with optional teacher forcing.

        Parameters
        ----------
        encoder_state : list of torch.Tensor
            One hidden state per encoder layer.
        horizon : int
            Number of future steps to predict.
        teacher : torch.Tensor or None
            Ground-truth targets of shape ``(B, horizon, N, output_dim)``.
            Only used when ``teacher_ratio > 0``.
        teacher_ratio : float
            Probability of feeding the ground-truth token at each step.
        """
        hidden_states = [s for s in encoder_state]
        batch, num_nodes = hidden_states[0].shape[:2]
        device = hidden_states[0].device
        outputs = []
        # First decoder input – a "go" token filled with the mean value.
        prev = torch.zeros(batch, num_nodes, self.output_dim, device=device)

        for t in range(horizon):
            layer_input = prev
            for i, cell in enumerate(self.cells):
                hidden_states[i] = cell(layer_input, hidden_states[i])
                layer_input = hidden_states[i]
            step_out = self.projection(hidden_states[-1])
            outputs.append(step_out)
            if teacher is not None and teacher_ratio > 0.0:
                use_teacher = torch.rand(1, device=device).item() < teacher_ratio
                prev = teacher[:, t] if use_teacher else step_out
            else:
                prev = step_out
        return torch.stack(outputs, dim=1)  # (B, horizon, N, output_dim)


class DCRNN(nn.Module):
    """End-to-end DCRNN encoder–decoder."""

    def __init__(
        self,
        supports: Sequence[torch.Tensor],
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_diffusion_step: int,
        horizon: int,
        cl_decay_steps: int = 2000,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.cl_decay_steps = cl_decay_steps
        self.encoder = DCRNNEncoder(
            supports, input_dim, hidden_dim, num_layers, max_diffusion_step
        )
        self.decoder = DCRNNDecoder(
            supports, output_dim, hidden_dim, num_layers, max_diffusion_step
        )

    def compute_teacher_ratio(self, global_step: int) -> float:
        """Scheduled sampling ratio – inverse sigmoid decay."""
        import math

        k = self.cl_decay_steps
        return float(k / (k + math.exp(global_step / k)))

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
        global_step: int | None = None,
    ) -> torch.Tensor:
        """Predict the next ``horizon`` speed values per sensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(B, T_in, N, F)``.
        targets : torch.Tensor, optional
            Shape ``(B, horizon, N, output_dim)`` – only used for
            scheduled sampling during training.
        global_step : int, optional
            Current optimiser step, controls the teacher ratio.
        """
        encoded = self.encoder(inputs)
        if self.training and targets is not None and global_step is not None:
            ratio = self.compute_teacher_ratio(global_step)
        else:
            ratio = 0.0
        return self.decoder(encoded, self.horizon, teacher=targets, teacher_ratio=ratio)
