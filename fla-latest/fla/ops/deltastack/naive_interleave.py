import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


def naive_interleave(
    current_pos: torch.Tensor,  # 3D: [B, T, H]
    prev_pos_global: torch.Tensor,  # 3D: [B, 1, H]
    v: torch.Tensor,  # 4D: [B, T, H, D]
    push_prob: torch.Tensor,  # 3D: [B, T, H]
    pop_prob: torch.Tensor,  # 3D: [B, T, H]
    stack_kappa: torch.Tensor,  # Scalar parameter
    stack_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unoptimized PyTorch reference.
    The future Triton kernel will take these exact same inputs and compute
    the Laplace distributions in registers to save memory bandwidth.
    """

    # --- 1. Compute Laplace (To be fused in Triton registers) ---
    grid_idx = torch.arange(
        stack_size, device=current_pos.device, dtype=torch.float32
    ).view(1, 1, 1, -1)
    sharpness = F.softplus(stack_kappa)

    def compute_laplace(ptr):
        p = ptr.unsqueeze(-1)
        dist = torch.abs(grid_idx - p)
        logits = -sharpness * dist
        return F.softmax(logits, dim=-1).to(dtype=v.dtype)

    k_stack = compute_laplace(current_pos)
    k_prev_global = compute_laplace(prev_pos_global)

    # --- 2. Interleave Logic (To be fused in Triton pointers) ---
    if cu_seqlens is not None:
        k_shifted = torch.roll(k_stack, shifts=1, dims=1)
        start_indices = cu_seqlens[:-1]
        k_prev_flat = k_prev_global.squeeze(1)
        k_shifted[0, start_indices] = k_prev_flat
        k_pop = k_shifted
    else:
        if k_stack.shape[1] > 1:
            k_pop = torch.cat([k_prev_global, k_stack[:, :-1]], dim=1)
        else:
            k_pop = k_prev_global

    v_pop = torch.zeros_like(v)
    beta_pop = pop_prob.unsqueeze(-1)

    k_push = k_stack
    v_push = v
    beta_push = push_prob.unsqueeze(-1)

    def interleave(a, b):
        combined = torch.stack([a, b], dim=2)
        return combined.flatten(1, 2)

    k_interleaved = interleave(k_pop, k_push).contiguous()
    v_interleaved = interleave(v_pop, v_push).contiguous()
    beta_interleaved = interleave(beta_pop, beta_push).squeeze(-1).contiguous()

    return k_interleaved, v_interleaved, beta_interleaved
