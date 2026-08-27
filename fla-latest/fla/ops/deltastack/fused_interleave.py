# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    input_guard,
)


# -----------------------------------------------------------------------------
# FORWARD KERNEL
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BT": 64}, num_warps=2),
        triton.Config({"BT": 64}, num_warps=4),
        triton.Config({"BT": 128}, num_warps=4),
        triton.Config({"BT": 128}, num_warps=8),
    ],
    key=["S", "D"],
    **autotune_cache_kwargs,
)
@triton.jit
def fused_prep_fwd_kernel(
    pos_push,
    pos_pop,
    v,
    push_prob,
    pop_prob,
    k_out,
    v_out,
    beta_out,
    kappa_ptr,
    T,
    stride_pos_b,
    stride_pos_t,
    stride_pos_h,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_ko_b,
    stride_ko_t,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_t,
    stride_vo_h,
    stride_vo_d,
    stride_bo_b,
    stride_bo_t,
    stride_bo_h,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    # 1. Identify which block of data this program is working on
    # i_bh is the combined Batch and Head index. i_t is the Time chunk.
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    # 2. Setup the offsets for the Time, Stack, and Head dimensions
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T  # Mask to prevent out-of-bounds reads on the last chunk

    o_s = tl.arange(0, S)
    o_d = tl.arange(0, D)

    # 3. Calculate memory pointers for inputs
    # Pointers to the 3D scalar tensors [B, T, H]
    p_pos_push = pos_push + i_b * stride_pos_b + o_t * stride_pos_t + i_h * stride_pos_h
    p_pos_pop = pos_pop + i_b * stride_pos_b + o_t * stride_pos_t + i_h * stride_pos_h
    p_push_prob = (
        push_prob + i_b * stride_pos_b + o_t * stride_pos_t + i_h * stride_pos_h
    )
    p_pop_prob = pop_prob + i_b * stride_pos_b + o_t * stride_pos_t + i_h * stride_pos_h

    # Pointers to the 4D V tensor [B, T, H, D]
    p_v = (
        v
        + i_b * stride_v_b
        + o_t[:, None] * stride_v_t
        + i_h * stride_v_h
        + o_d[None, :] * stride_v_d
    )

    # 4. Load data into ultra-fast SRAM registers
    # We load pointers as float32 for maximum precision during the distance calculation
    b_pos_push = tl.load(p_pos_push, mask=m_t, other=0.0).to(tl.float32)
    b_pos_pop = tl.load(p_pos_pop, mask=m_t, other=0.0).to(tl.float32)
    b_push_prob = tl.load(p_push_prob, mask=m_t, other=0.0)
    b_pop_prob = tl.load(p_pop_prob, mask=m_t, other=0.0)
    b_v = tl.load(p_v, mask=m_t[:, None], other=0.0)

    # Load scalar kappa and compute sharpness = softplus(kappa)
    kappa = tl.load(kappa_ptr).to(tl.float32)
    sharpness = tl.log(1.0 + tl.exp(kappa))

    # =====================================================================
    # THE MAGIC: Compute Laplace entirely in SRAM (No 4D Tensor written to HBM yet)
    # =====================================================================
    s_grid = o_s[None, :].to(tl.float32)  # Shape: [1, S]

    # -- Compute Push (Odd steps) --
    dist_push = tl.abs(s_grid - b_pos_push[:, None])  # Shape: [BT, S]
    logits_push = -sharpness * dist_push

    # Robust Softmax logic (subtract max for numerical stability)
    max_push = tl.max(logits_push, axis=1)
    exp_push = tl.exp(logits_push - max_push[:, None])
    sum_push = tl.sum(exp_push, axis=1)
    k_push = exp_push / sum_push[:, None]

    # -- Compute Pop (Even steps) --
    dist_pop = tl.abs(s_grid - b_pos_pop[:, None])  # Shape: [BT, S]
    logits_pop = -sharpness * dist_pop

    max_pop = tl.max(logits_pop, axis=1)
    exp_pop = tl.exp(logits_pop - max_pop[:, None])
    sum_pop = tl.sum(exp_pop, axis=1)
    k_pop = exp_pop / sum_pop[:, None]

    # =====================================================================
    # INTERLEAVED WRITING: Write directly to the output buffer
    # =====================================================================
    # Map the original time index `o_t` to the doubled interleaved timeline
    t_even = o_t * 2  # Pop steps
    t_odd = o_t * 2 + 1  # Push steps

    # Calculate output pointers
    # K_out shape: [B, 2T, H, S]
    p_k_even = (
        k_out
        + i_b * stride_ko_b
        + t_even[:, None] * stride_ko_t
        + i_h * stride_ko_h
        + o_s[None, :] * stride_ko_s
    )
    p_k_odd = (
        k_out
        + i_b * stride_ko_b
        + t_odd[:, None] * stride_ko_t
        + i_h * stride_ko_h
        + o_s[None, :] * stride_ko_s
    )

    # V_out shape: [B, 2T, H, D]
    p_v_even = (
        v_out
        + i_b * stride_vo_b
        + t_even[:, None] * stride_vo_t
        + i_h * stride_vo_h
        + o_d[None, :] * stride_vo_d
    )
    p_v_odd = (
        v_out
        + i_b * stride_vo_b
        + t_odd[:, None] * stride_vo_t
        + i_h * stride_vo_h
        + o_d[None, :] * stride_vo_d
    )

    # Beta_out shape: [B, 2T, H]
    p_beta_even = (
        beta_out + i_b * stride_bo_b + t_even * stride_bo_t + i_h * stride_bo_h
    )
    p_beta_odd = beta_out + i_b * stride_bo_b + t_odd * stride_bo_t + i_h * stride_bo_h

    # Execute memory stores
    tl.store(p_k_even, k_pop.to(k_out.dtype.element_ty), mask=m_t[:, None])
    tl.store(p_k_odd, k_push.to(k_out.dtype.element_ty), mask=m_t[:, None])

    tl.store(p_beta_even, b_pop_prob.to(beta_out.dtype.element_ty), mask=m_t)
    tl.store(p_beta_odd, b_push_prob.to(beta_out.dtype.element_ty), mask=m_t)

    # V is zero on pop, V on push
    b_zeros_v = tl.zeros([BT, D], dtype=tl.float32)
    tl.store(p_v_even, b_zeros_v.to(v_out.dtype.element_ty), mask=m_t[:, None])
    tl.store(p_v_odd, b_v.to(v_out.dtype.element_ty), mask=m_t[:, None])


# -----------------------------------------------------------------------------
# BACKWARD KERNEL (Crucial for Training)
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BT": 64}, num_warps=2),
        triton.Config({"BT": 64}, num_warps=4),
    ],
    key=["S", "D"],
    **autotune_cache_kwargs,
)
@triton.jit
def fused_prep_bwd_kernel(
    dk_in,
    dv_in,
    dbeta_in,  # Incoming gradients from chunk_delta_rule
    pos_push,
    pos_pop,
    kappa_ptr,  # Forward pass saved inputs
    k_out,  # Forward pass saved K (probabilities)
    dpos_push,
    dpos_pop,
    dv_out,
    dkappa_out,  # Output gradients
    T,
    stride_ko_b,
    stride_ko_t,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_t,
    stride_vo_h,
    stride_vo_d,
    stride_pos_b,
    stride_pos_t,
    stride_pos_h,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_s = tl.arange(0, S)
    o_d = tl.arange(0, D)

    # 1. Load Forward Pass Context
    kappa = tl.load(kappa_ptr).to(tl.float32)
    sharpness = tl.log(1.0 + tl.exp(kappa))
    sigmoid_kappa = 1.0 / (1.0 + tl.exp(-kappa))  # Derivative of softplus(kappa)

    p_pos_push = pos_push + i_b * stride_pos_b + o_t * stride_pos_t + i_h * stride_pos_h
    p_pos_pop = pos_pop + i_b * stride_pos_b + o_t * stride_pos_t + i_h * stride_pos_h
    b_pos_push = tl.load(p_pos_push, mask=m_t, other=0.0).to(tl.float32)
    b_pos_pop = tl.load(p_pos_pop, mask=m_t, other=0.0).to(tl.float32)

    # Re-calculate distances
    s_grid = o_s[None, :].to(tl.float32)
    dist_push = tl.abs(s_grid - b_pos_push[:, None])
    dist_pop = tl.abs(s_grid - b_pos_pop[:, None])

    # 2. Load incoming Gradients and Probabilities
    t_even = o_t * 2
    t_odd = o_t * 2 + 1

    # Load dK
    p_dk_even = (
        dk_in
        + i_b * stride_ko_b
        + t_even[:, None] * stride_ko_t
        + i_h * stride_ko_h
        + o_s[None, :] * stride_ko_s
    )
    p_dk_odd = (
        dk_in
        + i_b * stride_ko_b
        + t_odd[:, None] * stride_ko_t
        + i_h * stride_ko_h
        + o_s[None, :] * stride_ko_s
    )
    b_dk_pop = tl.load(p_dk_even, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_dk_push = tl.load(p_dk_odd, mask=m_t[:, None], other=0.0).to(tl.float32)

    # Load K (computed during forward pass, needed for softmax derivative)
    p_k_even = (
        k_out
        + i_b * stride_ko_b
        + t_even[:, None] * stride_ko_t
        + i_h * stride_ko_h
        + o_s[None, :] * stride_ko_s
    )
    p_k_odd = (
        k_out
        + i_b * stride_ko_b
        + t_odd[:, None] * stride_ko_t
        + i_h * stride_ko_h
        + o_s[None, :] * stride_ko_s
    )
    b_k_pop = tl.load(p_k_even, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_k_push = tl.load(p_k_odd, mask=m_t[:, None], other=0.0).to(tl.float32)

    # 3. Softmax Backward Math: dL_i = P_i * (dK_i - sum(P_j * dK_j))
    dot_push = tl.sum(b_k_push * b_dk_push, axis=1)
    dL_push = b_k_push * (b_dk_push - dot_push[:, None])

    dot_pop = tl.sum(b_k_pop * b_dk_pop, axis=1)
    dL_pop = b_k_pop * (b_dk_pop - dot_pop[:, None])

    # 4. Chain Rule for Pointers
    # d(dist)/d(pos) = sign(pos - grid)
    sign_push = tl.where(b_pos_push[:, None] > s_grid, 1.0, -1.0)
    sign_pop = tl.where(b_pos_pop[:, None] > s_grid, 1.0, -1.0)

    # dL / d(pos) = sum( dL_s * (-sharpness) * sign(pos - s) )
    b_dpos_push = tl.sum(dL_push * (-sharpness) * sign_push, axis=1)
    b_dpos_pop = tl.sum(dL_pop * (-sharpness) * sign_pop, axis=1)

    # 5. Chain Rule for Kappa
    # dL / d(sharpness) = sum( dL_s * (-dist_s) )
    dsharpness_push = tl.sum(dL_push * (-dist_push))
    dsharpness_pop = tl.sum(dL_pop * (-dist_pop))
    b_dkappa = (dsharpness_push + dsharpness_pop) * sigmoid_kappa

    # 6. Store Output Gradients
    p_dpos_push = (
        dpos_push + i_b * stride_pos_b + o_t * stride_pos_t + i_h * stride_pos_h
    )
    p_dpos_pop = dpos_pop + i_b * stride_pos_b + o_t * stride_pos_t + i_h * stride_pos_h

    tl.store(p_dpos_push, b_dpos_push.to(dpos_push.dtype.element_ty), mask=m_t)
    tl.store(p_dpos_pop, b_dpos_pop.to(dpos_pop.dtype.element_ty), mask=m_t)

    # We use atomic add for dkappa because all blocks update the same scalar
    tl.atomic_add(dkappa_out, b_dkappa)

    # De-interleave V (Odd indices contain the V gradients we care about)
    p_dv_odd = (
        dv_in
        + i_b * stride_vo_b
        + t_odd[:, None] * stride_vo_t
        + i_h * stride_vo_h
        + o_d[None, :] * stride_vo_d
    )
    b_dv = tl.load(p_dv_odd, mask=m_t[:, None], other=0.0)
    p_dv_out = (
        dv_out
        + i_b * stride_v_b
        + o_t[:, None] * stride_v_t
        + i_h * stride_v_h
        + o_d[None, :] * stride_v_d
    )
    tl.store(p_dv_out, b_dv.to(dv_out.dtype.element_ty), mask=m_t[:, None])


# -----------------------------------------------------------------------------
# PYTORCH AUTOGRAD WRAPPER
# -----------------------------------------------------------------------------
class FusedDeltastackPrepFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, pos_push, pos_pop, v, push_prob, pop_prob, stack_kappa, stack_size
    ):
        B, T, H = pos_push.shape
        D = v.shape[-1]
        S = stack_size

        # Pre-allocate contiguous interleaved outputs in HBM
        k_out = torch.empty((B, 2 * T, H, S), dtype=v.dtype, device=v.device)
        v_out = torch.empty((B, 2 * T, H, D), dtype=v.dtype, device=v.device)
        beta_out = torch.empty((B, 2 * T, H), dtype=v.dtype, device=v.device)

        grid = lambda META: (triton.cdiv(T, META["BT"]), B * H)

        fused_prep_fwd_kernel[grid](
            pos_push,
            pos_pop,
            v,
            push_prob,
            pop_prob,
            k_out,
            v_out,
            beta_out,
            stack_kappa,
            T,
            pos_push.stride(0),
            pos_push.stride(1),
            pos_push.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            k_out.stride(3),
            v_out.stride(0),
            v_out.stride(1),
            v_out.stride(2),
            v_out.stride(3),
            beta_out.stride(0),
            beta_out.stride(1),
            beta_out.stride(2),
            B=B,
            H=H,
            S=S,
            D=D,
        )

        ctx.save_for_backward(pos_push, pos_pop, stack_kappa, k_out)
        return k_out, v_out, beta_out

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dk_in, dv_in, dbeta_in):
        pos_push, pos_pop, stack_kappa, k_out = ctx.saved_tensors
        B, T, H = pos_push.shape
        S, D = k_out.shape[-1], dv_in.shape[-1]

        dpos_push = torch.empty_like(pos_push)
        dpos_pop = torch.empty_like(pos_pop)
        dv_out = torch.empty_like(dv_in[:, :T, ...])  # Shape [B, T, H, D]
        dkappa_out = torch.zeros_like(stack_kappa)  # Atomic add needs zeros

        grid = lambda META: (triton.cdiv(T, META["BT"]), B * H)

        fused_prep_bwd_kernel[grid](
            dk_in,
            dv_in,
            dbeta_in,
            pos_push,
            pos_pop,
            stack_kappa,
            k_out,
            dpos_push,
            dpos_pop,
            dv_out,
            dkappa_out,
            T,
            dk_in.stride(0),
            dk_in.stride(1),
            dk_in.stride(2),
            dk_in.stride(3),
            dv_in.stride(0),
            dv_in.stride(1),
            dv_in.stride(2),
            dv_in.stride(3),
            pos_push.stride(0),
            pos_push.stride(1),
            pos_push.stride(2),
            dv_out.stride(0),
            dv_out.stride(1),
            dv_out.stride(2),
            dv_out.stride(3),
            B=B,
            H=H,
            S=S,
            D=D,
        )

        # De-interleave Beta gradients directly via views (very fast, no kernel needed)
        dpop_prob = dbeta_in[:, 0::2, :]
        dpush_prob = dbeta_in[:, 1::2, :]

        return dpos_push, dpos_pop, dv_out, dpush_prob, dpop_prob, dkappa_out, None


# The main entrypoint you will call
def fused_interleave(
    current_pos: torch.Tensor,
    prev_pos_global: torch.Tensor,
    v: torch.Tensor,
    push_prob: torch.Tensor,
    pop_prob: torch.Tensor,
    stack_kappa: torch.Tensor,
    stack_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # 1. PyTorch pre-processing: Calculate pos_pop
    # We do this here so the Triton kernel is branchless and insanely fast.
    if cu_seqlens is not None:
        pos_pop = torch.roll(current_pos, shifts=1, dims=1)
        start_indices = cu_seqlens[:-1]
        pos_pop[0, start_indices] = prev_pos_global.squeeze(1)
    else:
        if current_pos.shape[1] > 1:
            pos_pop = torch.cat([prev_pos_global, current_pos[:, :-1]], dim=1)
        else:
            pos_pop = prev_pos_global

    # 2. Call the highly optimized Triton Autograd block
    return FusedDeltastackPrepFunction.apply(
        current_pos, pos_pop, v, push_prob, pop_prob, stack_kappa, stack_size
    )
