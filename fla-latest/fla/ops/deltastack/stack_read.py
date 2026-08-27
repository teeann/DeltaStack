# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


# -----------------------------------------------------------------------------
# FORWARD KERNEL
# -----------------------------------------------------------------------------
@triton.jit
def v_aug_fwd_kernel(
    v_ptr,
    s2x_ptr,
    prev_s_ptr,
    start_mask_ptr,
    v_aug_ptr,
    T,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_s2x_b,
    stride_s2x_t,
    stride_s2x_h,
    stride_s2x_d,
    stride_ps_b,
    stride_ps_h,
    stride_ps_d,
    stride_sm_b,
    stride_sm_t,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_PREV: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    # Map each block to a single Batch/Head/Time coordinate
    # We load the entire D dimension (which is small, 64 or 128) in one block
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_d = tl.arange(0, BLOCK_D)
    m_d = o_d < D  # Mask to prevent reading out of bounds if D isn't a power of 2

    # 1. Load V at current time `t`
    p_v = (
        v_ptr
        + i_b * stride_v_b
        + i_t * stride_v_t
        + i_h * stride_v_h
        + o_d * stride_v_d
    )
    b_v = tl.load(p_v, mask=m_d, other=0.0).to(tl.float32)

    # 2. Determine boundary logic (Is this the start of a sequence?)
    is_start = False
    if i_t == 0:
        is_start = True
    elif HAS_MASK:
        # Load from the start mask generated in PyTorch
        p_sm = start_mask_ptr + i_b * stride_sm_b + i_t * stride_sm_t
        mask_val = tl.load(p_sm)
        is_start = mask_val == 1.0

    # 3. Load the correct historical state (stack_read)
    if is_start:
        if HAS_PREV and i_t == 0:
            # Load from Cache
            p_ps = (
                prev_s_ptr + i_b * stride_ps_b + i_h * stride_ps_h + o_d * stride_ps_d
            )
            b_read = tl.load(p_ps, mask=m_d, other=0.0).to(tl.float32)
        else:
            # Zero-init
            b_read = tl.zeros([BLOCK_D], dtype=tl.float32)
    else:
        # Load from stack_out_2x at t-1.
        # The push step of t-1 is located at time index 2*(t-1) + 1
        t_s2x = 2 * (i_t - 1) + 1
        p_s2x = (
            s2x_ptr
            + i_b * stride_s2x_b
            + t_s2x * stride_s2x_t
            + i_h * stride_s2x_h
            + o_d * stride_s2x_d
        )
        b_read = tl.load(p_s2x, mask=m_d, other=0.0).to(tl.float32)

    # 4. Compute Subtraction and Store
    b_out = b_v - b_read
    p_out = (
        v_aug_ptr
        + i_b * stride_v_b
        + i_t * stride_v_t
        + i_h * stride_v_h
        + o_d * stride_v_d
    )
    tl.store(p_out, b_out.to(v_aug_ptr.dtype.element_ty), mask=m_d)


# -----------------------------------------------------------------------------
# BACKWARD KERNEL
# -----------------------------------------------------------------------------
@triton.jit
def v_aug_bwd_kernel(
    dv_aug_ptr,
    start_mask_ptr,
    ds2x_ptr,
    T,
    stride_dv_b,
    stride_dv_t,
    stride_dv_h,
    stride_dv_d,
    stride_ds_b,
    stride_ds_t,
    stride_ds_h,
    stride_ds_d,
    stride_sm_b,
    stride_sm_t,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_d = tl.arange(0, BLOCK_D)
    m_d = o_d < D

    # Because stack_read[t] = stack_out_2x[t-1],
    # the gradient for stack_out_2x[t] comes from dv_aug[t+1]
    next_t = i_t + 1
    b_grad = tl.zeros([BLOCK_D], dtype=tl.float32)

    if next_t < T:
        is_start = False
        if HAS_MASK:
            p_sm = start_mask_ptr + i_b * stride_sm_b + next_t * stride_sm_t
            mask_val = tl.load(p_sm)
            is_start = mask_val == 1.0

        # If t+1 is a new sequence, no gradient flows back to t
        if not is_start:
            p_dv = (
                dv_aug_ptr
                + i_b * stride_dv_b
                + next_t * stride_dv_t
                + i_h * stride_dv_h
                + o_d * stride_dv_d
            )
            # Derivative of (v - stack_read) w.r.t stack_read is -1
            b_grad = -tl.load(p_dv, mask=m_d, other=0.0).to(tl.float32)

    # Store directly into the odd index (push step) of dstack_out_2x
    t_s2x = 2 * i_t + 1
    p_ds2x = (
        ds2x_ptr
        + i_b * stride_ds_b
        + t_s2x * stride_ds_t
        + i_h * stride_ds_h
        + o_d * stride_ds_d
    )
    tl.store(p_ds2x, b_grad.to(ds2x_ptr.dtype.element_ty), mask=m_d)


# -----------------------------------------------------------------------------
# PYTORCH WRAPPER
# -----------------------------------------------------------------------------
class FusedVAugFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, v, stack_out_2x, prev_stack_out=None, cu_seqlens=None):
        B, T, H, D = v.shape
        v_aug = torch.empty_like(v)

        # Create a boolean mask in PyTorch for sequence boundaries (super fast)
        start_mask = None
        if cu_seqlens is not None:
            start_mask = torch.zeros((B, T), device=v.device, dtype=torch.float32)
            start_indices = cu_seqlens[:-1]
            start_mask[0, start_indices] = 1.0

        BLOCK_D = triton.next_power_of_2(D)
        grid = (T, B * H)

        HAS_PREV = prev_stack_out is not None
        HAS_MASK = start_mask is not None

        stride_ps_b = prev_stack_out.stride(0) if HAS_PREV else 0
        stride_ps_h = prev_stack_out.stride(1) if HAS_PREV else 0
        stride_ps_d = prev_stack_out.stride(2) if HAS_PREV else 0

        stride_sm_b = start_mask.stride(0) if HAS_MASK else 0
        stride_sm_t = start_mask.stride(1) if HAS_MASK else 0

        v_aug_fwd_kernel[grid](
            v,
            stack_out_2x,
            prev_stack_out,
            start_mask,
            v_aug,
            T,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            stack_out_2x.stride(0),
            stack_out_2x.stride(1),
            stack_out_2x.stride(2),
            stack_out_2x.stride(3),
            stride_ps_b,
            stride_ps_h,
            stride_ps_d,
            stride_sm_b,
            stride_sm_t,
            B=B,
            H=H,
            D=D,
            BLOCK_D=BLOCK_D,
            HAS_PREV=HAS_PREV,
            HAS_MASK=HAS_MASK,
        )

        ctx.save_for_backward(start_mask)
        ctx.HAS_PREV = HAS_PREV
        ctx.B, ctx.T, ctx.H, ctx.D = B, T, H, D
        return v_aug

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dv_aug):
        (start_mask,) = ctx.saved_tensors
        B, T, H, D = ctx.B, ctx.T, ctx.H, ctx.D

        # dv_aug passes cleanly to v (derivative of v is 1)
        dv = dv_aug

        # We must return dstack_out_2x of shape [B, 2T, H, D]
        # Pop steps (even indices) have zero gradient, so we init with zeros
        dstack_out_2x = torch.zeros(
            (B, 2 * T, H, D), dtype=dv_aug.dtype, device=dv_aug.device
        )

        HAS_MASK = start_mask is not None
        stride_sm_b = start_mask.stride(0) if HAS_MASK else 0
        stride_sm_t = start_mask.stride(1) if HAS_MASK else 0

        BLOCK_D = triton.next_power_of_2(D)
        grid = (T, B * H)

        v_aug_bwd_kernel[grid](
            dv_aug,
            start_mask,
            dstack_out_2x,
            T,
            dv_aug.stride(0),
            dv_aug.stride(1),
            dv_aug.stride(2),
            dv_aug.stride(3),
            dstack_out_2x.stride(0),
            dstack_out_2x.stride(1),
            dstack_out_2x.stride(2),
            dstack_out_2x.stride(3),
            stride_sm_b,
            stride_sm_t,
            B=B,
            H=H,
            D=D,
            BLOCK_D=BLOCK_D,
            HAS_MASK=HAS_MASK,
        )

        dprev_stack_out = None
        if ctx.HAS_PREV:
            # The gradient for the cache comes from the first timestep
            dprev_stack_out = -dv_aug[:, 0, :, :].unsqueeze(1)

        return dv, dstack_out_2x, dprev_stack_out, None


# Main API to call from deltastack.py
def fused_v_aug(v, stack_out_2x, prev_stack_out=None, cu_seqlens=None, indices=None):
    # Process unpadded inference cache natively in PyTorch (fast and robust)
    p_stack_out = None
    if prev_stack_out is not None:
        if indices is not None:
            cache_flat = prev_stack_out.squeeze(1)
            p_stack_out = cache_flat[indices]
        else:
            p_stack_out = prev_stack_out.squeeze(1)

    return FusedVAugFunction.apply(v, stack_out_2x, p_stack_out, cu_seqlens)
