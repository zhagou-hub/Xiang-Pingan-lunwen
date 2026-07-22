"""ULA-Net model (ported and cleaned from the authors' ULA.py)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish / SiLU-style activation: f(h) = h * sigmoid(beta * h)."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class GatedLinearUnit(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.gate = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) * torch.sigmoid(self.gate(x))


class UnidirectionalAttention(nn.Module):
    """
    Unidirectional local multi-head attention over a KxK patch.

    Input shape: (B, L, K, K). The center pixel is the query; neighbors are keys/values.
    The center is masked in attention so spatial context comes from neighbors only,
    then DFS-style feature fusion mixes center spectral value with attended spatial context.
    """

    def __init__(
        self,
        n_bands: int,
        patch_size: int,
        n_heads: int = 4,
        val_dim: int = 128,
        key_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.patch_size = patch_size
        assert n_heads == 0 or (val_dim % n_heads == 0 and key_dim % n_heads == 0)
        self.head_num = n_heads
        heads = 1 if n_heads == 0 else n_heads
        self.val_split = val_dim // heads
        self.key_split = key_dim // heads
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.seq_len = patch_size * patch_size
        self.seq_center = self.seq_len // 2

        central_mask = torch.zeros(1, 1, self.seq_len)
        central_mask[..., self.seq_center] = -1e6
        self.register_buffer("central_mask", central_mask, persistent=False)

        self.softmax = nn.Softmax(dim=2)
        self.w_key = nn.Linear(n_bands, key_dim)
        self.w_val = nn.Linear(n_bands, val_dim)
        self.w_que = nn.Linear(n_bands, key_dim)
        self.nearby_gate = nn.Sequential(nn.Linear(val_dim, 1), nn.Sigmoid())
        self.last_gate: torch.Tensor | None = None
        self._init_weight()

    def _init_weight(self) -> None:
        for param in self.parameters():
            if param.ndim == 2:
                stdv = 1.0 / param.shape[1] ** 0.5
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, K, K) -> (B, KK, L)
        x = x.view(x.shape[0], x.shape[1], self.seq_len).transpose(1, 2)
        keys = self.w_key(x)
        vals = self.w_val(x)
        query = self.w_que(x[:, self.seq_center : self.seq_center + 1])

        attended_val = []
        for head_idx in range(self.head_num):
            small_key = keys[..., head_idx * self.key_split : (head_idx + 1) * self.key_split]
            small_query = query[..., head_idx * self.key_split : (head_idx + 1) * self.key_split]
            small_atten_weight = small_query @ small_key.transpose(1, 2) / np.sqrt(self.key_dim)
            small_atten = self.softmax(small_atten_weight + self.central_mask)
            small_val = vals[..., head_idx * self.val_split : (head_idx + 1) * self.val_split]
            attended_val.append(small_atten @ small_val)

        central_val = vals[:, self.seq_center : self.seq_center + 1]
        if self.head_num == 0:
            self.last_gate = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            return central_val

        surround_val = torch.cat(attended_val, dim=2)
        gate = self.nearby_gate(central_val - surround_val)
        self.last_gate = gate.view(gate.shape[0], -1).mean(dim=1, keepdim=True)
        return central_val + gate * surround_val


class RuaAE(nn.Module):
    """
    ULA-Net autoencoder with ELMM scaling decoder.

    Parameters
    ----------
    n_bands : L
    n_endmembers : P
    patch_size : K
    endmembers : initial endmember matrix, shape (P, L) to match Linear-style matmul,
                 or (L, P). Internally stored as Parameter of shape (P, L).
    """

    def __init__(
        self,
        n_bands: int,
        n_endmembers: int,
        patch_size: int,
        endmembers: torch.Tensor,
        model_dim: int = 128,
        n_heads: int = 4,
        scale_range: float = 0.2,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        self.patch_size = patch_size
        self.scale_range = scale_range

        edm = endmembers.detach().float()
        if edm.ndim != 2:
            raise ValueError(f"endmembers must be 2-D, got shape {tuple(edm.shape)}")
        if edm.shape == (n_bands, n_endmembers):
            edm = edm.T.contiguous()
        elif edm.shape != (n_endmembers, n_bands):
            raise ValueError(
                f"endmembers shape {tuple(edm.shape)} incompatible with "
                f"(P,L)=({n_endmembers},{n_bands}) or (L,P)"
            )
        self.endmember = nn.Parameter(edm.clamp(min=0.0))

        self.uda = UnidirectionalAttention(
            n_bands,
            patch_size,
            n_heads=n_heads,
            key_dim=model_dim,
            val_dim=model_dim,
        )
        self.ffn = nn.Sequential(
            Swish(),
            GatedLinearUnit(model_dim, 64),
            nn.BatchNorm1d(64),
            Swish(),
            GatedLinearUnit(64, n_endmembers),
            nn.Softmax(dim=1),
        )
        self.scalar = nn.Sequential(
            Swish(),
            GatedLinearUnit(model_dim, 64),
            nn.BatchNorm1d(64),
            Swish(),
            GatedLinearUnit(64, n_endmembers),
            nn.Tanh(),
        )
        self._init_weight()

    def _init_weight(self) -> None:
        for name, param in self.named_parameters():
            if name == "endmember":
                continue
            if param.ndim == 2:
                stdv = 1.0 / param.shape[1] ** 0.5
                nn.init.uniform_(param, -stdv, stdv)

    def projected_endmembers(self) -> torch.Tensor:
        """Paper constraint: endmembers are non-negative reflectance."""
        return self.endmember.clamp(min=0.0)

    def forward(
        self,
        x: torch.Tensor,
        return_abd: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ss_feat = self.uda(x).squeeze(1)
        abd = self.ffn(ss_feat)
        edm_scale = self.scalar(ss_feat)
        scale = 1.0 + self.scale_range * edm_scale.unsqueeze(-1)
        # ELMM: per-pixel scaled non-negative endmembers
        edm = scale * self.projected_endmembers().unsqueeze(0)
        recon = (abd.unsqueeze(1) @ edm).squeeze(1)

        if return_abd:
            return recon, abd, edm_scale
        return recon

    @torch.no_grad()
    def project_parameters_(self) -> None:
        """Project learnable endmembers onto the non-negative orthant after each step."""
        self.endmember.clamp_(min=0.0)

    @property
    def gate_values(self) -> torch.Tensor | None:
        return self.uda.last_gate
