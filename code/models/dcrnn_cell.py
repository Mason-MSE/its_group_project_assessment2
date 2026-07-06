"""Diffusion Convolutional GRU (DCGRU) cell.

The DCGRU cell replaces the linear operations inside a GRU with a
bidirectional random-walk diffusion convolution, so that information
from upstream and downstream neighbours is aggregated at every time
step.  For an input signal ``H`` and filter parameter ``\\Theta``, the
K-step diffusion convolution is

.. math::

    (\\Theta *_{G} H) = \\sum_{k=0}^{K-1}
        \\left( \\theta_{k,1}\\, (D_{O}^{-1} A)^{k}
              + \\theta_{k,2}\\, (D_{I}^{-1} A^{T})^{k} \\right) H,

which reduces to a standard MLP when ``K = 0``.

This module implements the cell in PyTorch and is imported by
:mod:`models.dcrnn`.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DiffusionGraphConv(nn.Module):
    """K-step bidirectional random-walk diffusion convolution.

    Parameters
    ----------
    supports : Sequence of torch.Tensor
        The random-walk transition matrices.  Two supports are typical:
        forward (``D_O^{-1} A``) and backward (``D_I^{-1} A^T``).
    input_dim, output_dim : int
        Feature dimensionality before and after the convolution.
    max_diffusion_step : int
        The order ``K`` of the diffusion.  Practical values are 1–3.
    bias : bool
        Whether to add a learnable bias term.
    """

    def __init__(
        self,
        supports: Sequence[torch.Tensor],
        input_dim: int,
        output_dim: int,
        max_diffusion_step: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if max_diffusion_step < 1:
            raise ValueError("max_diffusion_step must be >= 1")
        self.max_diffusion_step = max_diffusion_step
        self.num_supports = len(supports)
        # supports are registered as buffers so they move with the module.
        for i, s in enumerate(supports):
            self.register_buffer(f"support_{i}", s)

        # Number of independent filters:
        #   1 (identity) + max_diffusion_step * num_supports
        num_matrices = 1 + max_diffusion_step * self.num_supports
        self.weight = nn.Parameter(
            torch.empty(input_dim * num_matrices, output_dim)
        )
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter("bias", None)

    def _get_supports(self) -> Sequence[torch.Tensor]:
        return [getattr(self, f"support_{i}") for i in range(self.num_supports)]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the diffusion convolution.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(batch, num_nodes, input_dim)``.

        Returns
        -------
        out : torch.Tensor
            Shape ``(batch, num_nodes, output_dim)``.
        """
        batch, num_nodes, input_dim = inputs.shape
        # Start with the identity term x0 = H.
        x0 = inputs  # (B, N, F)
        collected = [x0]
        for support in self._get_supports():
            x_k_minus_1 = x0
            for _ in range(self.max_diffusion_step):
                # (N, N) x (B, N, F) -> (B, N, F) via einsum on the node axis
                x_k = torch.einsum("nm,bmf->bnf", support, x_k_minus_1)
                collected.append(x_k)
                x_k_minus_1 = x_k
        # Concatenate across the feature axis and apply the linear map.
        # Shape: (B, N, F * num_matrices) -> (B, N, output_dim)
        concat = torch.cat(collected, dim=-1)
        out = concat @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class DCGRUCell(nn.Module):
    """Diffusion-Convolutional GRU cell.

    Follows Li et al. (2018).  Reset gate, update gate and candidate
    state are all produced by diffusion convolutions rather than
    ordinary linear layers.
    """

    def __init__(
        self,
        supports: Sequence[torch.Tensor],
        input_dim: int,
        hidden_dim: int,
        max_diffusion_step: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        combined = input_dim + hidden_dim

        # Two "gate" convolutions produce reset and update in one call.
        self.gate_conv = DiffusionGraphConv(
            supports, combined, 2 * hidden_dim, max_diffusion_step
        )
        # Candidate state convolution.
        self.candidate_conv = DiffusionGraphConv(
            supports, combined, hidden_dim, max_diffusion_step
        )

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Return the next hidden state.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(batch, num_nodes, input_dim)``.
        hidden : torch.Tensor
            Shape ``(batch, num_nodes, hidden_dim)``.
        """
        combined = torch.cat([inputs, hidden], dim=-1)
        gate = torch.sigmoid(self.gate_conv(combined))
        r, u = torch.split(gate, self.hidden_dim, dim=-1)
        combined_r = torch.cat([inputs, r * hidden], dim=-1)
        c = torch.tanh(self.candidate_conv(combined_r))
        new_hidden = u * hidden + (1.0 - u) * c
        return new_hidden

    def init_hidden(self, batch_size: int, num_nodes: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
