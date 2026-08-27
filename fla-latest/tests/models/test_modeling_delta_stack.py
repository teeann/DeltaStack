# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers import DeltaStack
from fla.models import DeltaStackConfig

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
#
# run_test_model_forward_backward also
# covers fixed-batch vs. varlen (cu_seqlens) consistency internally.
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-use_l2warp{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    dtype: torch.dtype,
):
    run_test_model_forward_backward(L, B, T, H, D, DeltaStackConfig, use_l2warp=use_l2warp, dtype=dtype)


@pytest.mark.parametrize("stack_size", [4, 8, 32])
def test_modeling_stack_sizes(stack_size: int):
    """stack_size is the whole point of the layer, so exercise more than the
    default: it sets the depth of the bounded stack memory and therefore the
    shape of the carried stack state."""
    run_test_model_forward_backward(
        2, 2, 256, 4, 64, DeltaStackConfig, use_l2warp=False, dtype=torch.bfloat16,
        stack_size=stack_size,
    )


# ===================================================================================
# Test for Generation (K/V cache)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    run_test_generation(L, B, T, H, D, DeltaStackConfig, dtype, tol=8e-3)


# ===================================================================================
# Layer-level tests specific to the stack machinery
# ===================================================================================
def test_layer_shapes_and_backward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    layer = DeltaStack(hidden_size=128, num_heads=4, stack_size=8, layer_idx=0).to(device).bfloat16()
    x = torch.randn(2, 128, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    out, _, _ = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    # The action head drives push/pop; if it were detached from the graph the
    # stack pointer could never learn, so check that specific gradient exists.
    assert layer.action_proj.weight.grad is not None
    assert torch.isfinite(layer.action_proj.weight.grad).all()


@pytest.mark.parametrize("T", [96, 128, 1024])
def test_varlen_matches_fixed_batch(T: int):
    """Packed varlen must equal the fixed-batch result exactly.

    Segments are kept above 64 deliberately: the layer selects
    `fused_recurrent` when q_len <= 64 (delta_stack.py's `mode` line), so a
    short fixed batch would run fused_recurrent while the packed call (whose
    q_len is the sum) runs chunk, and the comparison would be between two
    kernels rather than two layouts. At T=64 that shows up as a ~1.6e-2 diff
    on activations of scale ~2.2 -- a pre-existing property shared with
    the stack addressing, not a packing bug.

    T=1024 is here because the short cases are not sufficient: the pointer
    cumsum's rounding drift grows with sequence length and compounds across
    stacked layers, so a version of this layer can pass at T=96/128 and still
    fail the 4-layer T=1024 modeling test. That is exactly what happened while
    making the varlen path compilable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    layer = DeltaStack(hidden_size=128, num_heads=4, stack_size=8, layer_idx=0).to(device).bfloat16()
    B = 3
    x = torch.randn(B, T, 128, device=device, dtype=torch.bfloat16)
    out_fixed, _, _ = layer(x)
    cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)
    out_var, _, _ = layer(x.reshape(1, B * T, 128), cu_seqlens=cu_seqlens)
    assert torch.equal(out_fixed.reshape(1, B * T, -1), out_var), \
        f"max diff {(out_fixed.reshape(1, B * T, -1).float() - out_var.float()).abs().max().item():.3e}"


def test_cache_tuple_shape():
    """The carried state is a 4-tuple (assoc_state, stack_state, ptr_state,
    prev_stack_out). Pin the shapes so a future refactor cannot silently
    reorder or drop one."""
    from .test_modeling_utils import create_model_and_config
    model, config = create_model_and_config(
        DeltaStackConfig, L=2, H=4, D=64, dtype=torch.bfloat16, stack_size=8,
    )
    device = next(model.parameters()).device
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 96), device=device)
    out = model(input_ids, use_cache=True)
    state = out.past_key_values[0]["recurrent_state"]
    assert len(state) == 4
    assoc, stack, ptr, prev_out = state
    assert assoc.shape == (2, 4, 64, 64)      # [B, H, head_k, head_v]
    assert stack.shape == (2, 4, 8, 64)       # [B, H, stack_size, head_v]
    assert ptr.shape == (2, 4)                # [B, H] scalar pointer per head
    assert prev_out.shape == (2, 1, 4, 64)    # [B, 1, H, head_v]

    next_token = out.logits[:, -1:].argmax(dim=-1)
    out2 = model(next_token, past_key_values=out.past_key_values, use_cache=True)
    assert out2.logits.shape == (2, 1, config.vocab_size)


def test_set_stack_size():
    """train.py's set_model_stack_size/reset_model_stack_size drive these at
    eval time, so the stack can be resized for evaluation."""
    layer = DeltaStack(hidden_size=128, num_heads=4, stack_size=8, layer_idx=0)
    assert layer.stack_size == 8
    layer.set_stack_size(32)
    assert layer.stack_size == 32
    layer.reset_stack_size()
    assert layer.stack_size == 8


def test_forward_survives_set_stack_size():
    """A full forward after set_stack_size must not crash -- this eval-time
    resize once crashed on a stale address dim."""
    from .test_modeling_utils import create_model_and_config
    model, config = create_model_and_config(
        DeltaStackConfig, L=2, H=4, D=64, dtype=torch.bfloat16, stack_size=8,
    )
    for module in model.modules():
        if isinstance(module, DeltaStack):
            module.set_stack_size(32)
    device = next(model.parameters()).device
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 96), device=device)
    out = model(input_ids, use_cache=True)
    assert out.logits.shape == (2, 96, config.vocab_size)
    state = out.past_key_values[0]["recurrent_state"]
    # The stack state must follow the new size, not the constructor default.
    assert state[1].shape == (2, 4, 32, 64)
