"""Losses aligned with the paper: SSD + L1/2 sparsity + gate regularization.

Also includes an optional MSE term. Pure SSD is scale-invariant; combined with
Softmax + L1/2 it can collapse abundances to near one-hot. A small MSE term
restores amplitude fidelity under the ELMM decoder (same physical model as the paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def squared_sine_distance(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Squared sine spectral distance averaged over the batch.

    sin^2(theta) = 1 - cos^2(theta), where theta is the spectral angle.
    """
    x = x.reshape(x.shape[0], -1)
    x_hat = x_hat.reshape(x_hat.shape[0], -1)
    cos = (x * x_hat).sum(dim=1) / (x.norm(dim=1) * x_hat.norm(dim=1) + eps)
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return (1.0 - cos.pow(2)).mean()


def half_norm_sparsity(abundance: torch.Tensor) -> torch.Tensor:
    """L_{1/2} sparsity: mean over batch of sum_j sqrt(|a_j|)."""
    return abundance.abs().clamp_min(1e-12).sqrt().sum(dim=1).mean()


class ULANetLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1e-1,
        beta: float = 1e-2,
        mse_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_weight = mse_weight

    def forward(
        self,
        target: torch.Tensor,
        recon: torch.Tensor,
        abundance: torch.Tensor,
        gate: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_re = squared_sine_distance(target, recon)
        loss_mse = F.mse_loss(recon, target)
        loss_sp = half_norm_sparsity(abundance)
        if gate is None:
            loss_gate = recon.new_tensor(0.0)
        else:
            loss_gate = gate.mean()

        # Paper: L = L_re + alpha*L_sp + beta*L_gate
        # Extra mse_weight*L_mse keeps abundances continuous under ELMM (amplitude).
        total = (
            loss_re
            + self.mse_weight * loss_mse
            + self.alpha * loss_sp
            + self.beta * loss_gate
        )
        stats = {
            "loss": float(total.detach().cpu()),
            "loss_re": float(loss_re.detach().cpu()),
            "loss_mse": float(loss_mse.detach().cpu()),
            "loss_sparsity": float(loss_sp.detach().cpu()),
            "loss_gate": float(loss_gate.detach().cpu()),
        }
        return total, stats
