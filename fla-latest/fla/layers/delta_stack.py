# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
from fla.ops.deltastack import naive_interleave, fused_interleave, fused_v_aug

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache


def _warn_if_stack_size_not_pow2(stack_size: int):
    """The fused interleave kernels index the stack with `tl.arange(0, S)`,
    which Triton only accepts for a power-of-2 range -- a non-power-of-2
    stack size fails at kernel compile time, not here."""
    if stack_size is not None and stack_size > 0 and (stack_size & (stack_size - 1)) != 0:
        warnings.warn(
            f"DeltaStack stack_size={stack_size} is not a power of 2; the fused "
            "interleave kernel requires a power-of-2 stack size and will fail to "
            "compile with: arange's range must be a power of 2. Use "
            f"stack_size={1 << (stack_size - 1).bit_length()} instead.",
            stacklevel=3,
        )


def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class DeltaStack(nn.Module):
    r"""
    The layer implementaion for [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484).  # noqa:
    DeltaStack was originally proposed in [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174). # noqa
    """

    def __init__(
        self,
        mode: str = "chunk",
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        stack_size: int = 8,
        **kwargs,
    ) -> DeltaStack:
        super().__init__()

        _warn_if_stack_size_not_pow2(stack_size)
        self.stack_size = stack_size
        self.stack_kappa = nn.Parameter(torch.tensor([2.0]))  # for fully shard

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        assert self.qk_activation in ["silu", "relu", "elu", "identity"]
        assert self.qk_norm in ["l2", "sum"]

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        if mode == "fused_chunk":
            raise NotImplementedError(
                "fused_chunk_delta_rule is now deprecated. Please use `chunk_delta_rule` instead."
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."
        assert (
            self.key_dim % num_heads == 0
        ), f"key dim must be divisible by num_heads of {num_heads}"
        assert (
            self.value_dim % num_heads == 0
        ), f"value dim must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.action_proj = nn.Linear(hidden_size, self.num_heads * 3, bias=False)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu" if qk_activation == "silu" else None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu" if qk_activation == "silu" else None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.",
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _generate_soft_pointer(
        self,
        actions: torch.Tensor,
        prev_ptr: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the raw 3D pointers.
        Note: Laplace calculation is deferred to prepare_interleaved_tensors_with_laplace.
        """
        # 1. Decode Actions
        action_probs = F.softmax(actions.float(), dim=-1)
        push_prob = action_probs[..., 0]
        pop_prob = action_probs[..., 1]

        delta_ptr = push_prob - pop_prob

        # 2. Integrate (Get RAW Positions)
        # .shape[0] rather than len(): under torch.compile the latter is a Python
        # int that specializes the graph on the document count. Only used as a
        # tensor dimension here, so it need not specialize.
        bsz = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else actions.size(0)

        if cu_seqlens is not None:
            # FLATTENED (varlen) MODE -- single segmented cumsum, no Python loop.
            #
            # This replaced a `for i in range(bsz)` loop that sliced
            # cu_seqlens[i]:cu_seqlens[i+1] per document. That loop was correct
            # but made the layer effectively uncompilable: `len(cu_seqlens)`
            # hard-specializes the graph on the document count (which changes
            # every batch under varlen packing, so it recompiles constantly
            # and eventually trips
            # ConstraintViolationError against mark_dynamic), and `cu_seqlens[i]`
            # forces a Tensor.item() graph break. It also launched O(num_docs)
            # kernels instead of one.
            #
            # Everything below is shape-static in the document count: cumsum over
            # the whole packed buffer, then rebase each segment by subtracting the
            # prefix at its own start, so no document's pointer leaks into the next.
            T_len = delta_ptr.shape[1]
            num_seqs = cu_seqlens.shape[0] - 1

            if prev_ptr is None:
                prev_ptr = torch.zeros(
                    (num_seqs, self.num_heads),
                    dtype=torch.float32,
                    device=actions.device,
                )
            else:
                prev_ptr = prev_ptr.float()

            starts = cu_seqlens[:-1].long()
            is_start = torch.zeros(1, T_len, dtype=torch.bool, device=actions.device)
            is_start[0, starts] = True
            seg_id = is_start.long().cumsum(1) - 1  # [1, T] -> which document each position is in

            # fp64 for the accumulation: at fp32 a packed buffer of 32768 puts the
            # running sum where eps is ~2e-3, and the rebasing subtraction would
            # then make a document's pointer depend on how it happened to be
            # packed. The per-document loop this replaces did not have that
            # exposure (each segment started its cumsum at 0), so matching its
            # numerics is a correctness requirement, not a nicety.
            step = delta_ptr.double()
            cs = torch.cumsum(step, dim=1)
            seg_base = (cs - step)[0, starts]                      # [num_seqs, H]
            pointer_change = cs - seg_base[seg_id[0]].unsqueeze(0)  # [1, T, H]
            prev_pos_flat = prev_ptr[seg_id[0]].unsqueeze(0).double()

            current_pos = (prev_pos_flat + pointer_change).to(delta_ptr.dtype)

            # CACHE LOGIC: Extract UNCLAMPED last states
            last_indices = cu_seqlens[1:].long() - 1
            new_ptr_state_raw = current_pos[0, last_indices]

        else:
            # BATCH MODE
            # fp64 here too, so the two branches accumulate identically. At fp32
            # a 1024-step cumsum drifts enough that the batch and varlen results
            # separate by more than the 1e-3 the modeling test allows -- and the
            # drift belongs to *this* branch, since the varlen branch above is
            # exact. Keeping them on the same arithmetic is what makes
            # "packing must not change the answer" hold at long sequence length.
            pointer_change = torch.cumsum(delta_ptr.double(), dim=1)

            if prev_ptr is None:
                prev_ptr = torch.zeros(
                    (bsz, self.num_heads),
                    dtype=torch.float32,
                    device=actions.device,
                )
            else:
                prev_ptr = prev_ptr.float()

            prev_pos = prev_ptr.unsqueeze(1).double()
            current_pos = (prev_pos + pointer_change).to(delta_ptr.dtype)

            # CACHE LOGIC: Extract UNCLAMPED last state
            new_ptr_state_raw = current_pos[:, -1]

        # 3. Clamping for Kernel (Physical Memory Access)
        current_pos_clamped = torch.clamp(current_pos, 0, self.stack_size - 1.0)

        if prev_ptr is not None:
            prev_ptr_clamped = torch.clamp(prev_ptr, 0, self.stack_size - 1.0)
        else:
            prev_ptr_clamped = torch.zeros(
                (bsz, self.num_heads), dtype=torch.float32, device=actions.device
            )

        # Return 3D Pointers. Laplace expansion is deferred to the modular/Triton function.
        return (
            current_pos_clamped,
            push_prob.to(actions.dtype),
            pop_prob.to(actions.dtype),
            new_ptr_state_raw,
            prev_ptr_clamped.unsqueeze(1),  # [B, 1, H]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if q_len <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        recurrent_state, stack_recurrent_state, stack_ptr_state = None, None, None
        prev_stack_out = None
        if last_state is not None:
            recurrent_state, stack_recurrent_state, stack_ptr_state, prev_stack_out = (
                last_state["recurrent_state"]
            )

        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None

        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            if self.qk_activation == "silu":
                q, k = F.silu(q), F.silu(k)
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        if self.qk_activation != "silu":
            if self.qk_activation == "relu":
                q, k = q.relu(), k.relu()
            elif self.qk_activation == "elu":
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != "identity":
                raise NotImplementedError

        if self.qk_norm == "sum":
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])

        if self.allow_neg_eigval:
            beta = beta * 2.0

        # 4. Pointer Generation
        action_logits = self.action_proj(hidden_states)
        actions = rearrange(
            action_logits, "b t (h d) -> b t h d", h=self.num_heads, d=3
        )

        ptr_curr, push_prob, pop_prob, new_ptr_state, ptr_prev_global = (
            self._generate_soft_pointer(
                actions, prev_ptr=stack_ptr_state, cu_seqlens=cu_seqlens
            )
        )

        # === 6. INTERLEAVING VIA MODULAR FUNCTION ===
        k_interleaved, v_interleaved, beta_interleaved = (
            # naive_interleave(
            fused_interleave(
                current_pos=ptr_curr,
                prev_pos_global=ptr_prev_global,
                v=v,
                push_prob=push_prob,
                pop_prob=pop_prob,
                stack_kappa=self.stack_kappa,
                stack_size=self.stack_size,
                cu_seqlens=cu_seqlens,
            )
        )

        cu_seqlens_2x = None
        if cu_seqlens is not None:
            cu_seqlens_2x = cu_seqlens * 2

        if mode == "chunk":
            stack_out_2x, new_stack_recurrent_state = chunk_delta_rule(
                q=k_interleaved,
                k=k_interleaved,
                v=v_interleaved,
                beta=beta_interleaved,
                initial_state=stack_recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens_2x,
                # use_qk_l2norm_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            stack_out_2x, new_stack_recurrent_state = fused_recurrent_delta_rule(
                q=k_interleaved,
                k=k_interleaved,
                v=v_interleaved,
                beta=beta_interleaved,
                initial_state=stack_recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens_2x,
                # use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # === 8. TIME-SHIFT LOGIC (TRAIN vs DECODE) ===
        if q_len == 1:
            # Memory bloat only happens at large T. For autoregressive decoding (T=1),
            # we use pure PyTorch to perfectly handle varlen continuous batching edge cases.
            stack_out = stack_out_2x[:, 1::2].contiguous()
            
            if prev_stack_out is not None:
                if indices is not None:
                    cache_flat = prev_stack_out.squeeze(1)
                    stack_read = cache_flat[indices].unsqueeze(0)
                else:
                    stack_read = prev_stack_out
            else:
                stack_read = torch.zeros_like(stack_out)
                
            v_aug = v - stack_read
        else:
            # For training (large T), use the highly optimized memory-efficient kernel
            v_aug = fused_v_aug(v, stack_out_2x, prev_stack_out, cu_seqlens, indices)

        # === 9. ASSOCIATIVE MEMORY UPDATE ===
        if mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q,
                k=k,
                v=v_aug,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=(self.qk_norm == "l2"),
            )
        elif mode == "chunk":
            o, recurrent_state = chunk_delta_rule(
                q=q,
                k=k,
                v=v_aug,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=(self.qk_norm == "l2"),
            )

        # === 10. CACHE EXTRACTION ===
        next_prev_stack_out = None
        if use_cache:
            if cu_seqlens is not None:
                # The last push step of a sequence is the very last element of its doubled segment
                last_indices_2x = cu_seqlens_2x[1:] - 1
                # Extract, reshape, and FORCE contiguous memory to prevent downstream crashes
                next_prev_stack_out = stack_out_2x[0, last_indices_2x].unsqueeze(1).contiguous()
            else:
                # In standard batch mode, the very last element (-1) is the final push step
                next_prev_stack_out = stack_out_2x[:, -1:].contiguous()

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=(
                    recurrent_state,
                    new_stack_recurrent_state,
                    new_ptr_state,
                    next_prev_stack_out,
                ),
                conv_state=(
                    (conv_state_q, conv_state_k, conv_state_v)
                    if self.use_short_conv
                    else None
                ),
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values

    def set_stack_size(self, new_size: int):
        _warn_if_stack_size_not_pow2(new_size)
        self.orig_stack_size = self.stack_size
        self.stack_size = new_size

    def reset_stack_size(self):
        self.stack_size = self.orig_stack_size
