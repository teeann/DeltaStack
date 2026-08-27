# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Trimmed vendored fork: only the ops the kept layers use. Upstream fla ships
# many more.

from .attn import parallel_attn
from .delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_delta_rule
from .gated_delta_rule import chunk_gated_delta_rule, chunk_gdn, fused_recurrent_gated_delta_rule, fused_recurrent_gdn
from .generalized_delta_rule import (
    chunk_dplr_delta_rule,
    chunk_iplr_delta_rule,
    fused_recurrent_dplr_delta_rule,
    fused_recurrent_iplr_delta_rule,
)
from .gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from .rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
from .simple_gla import chunk_simple_gla, fused_chunk_simple_gla, fused_recurrent_simple_gla, parallel_simple_gla

__all__ = [
    'chunk_delta_rule',
    'chunk_dplr_delta_rule',
    'chunk_gated_delta_rule',
    'chunk_gdn',
    'chunk_gla',
    'chunk_iplr_delta_rule',
    'chunk_rwkv6',
    'chunk_simple_gla',
    'fused_chunk_delta_rule',
    'fused_chunk_gla',
    'fused_chunk_simple_gla',
    'fused_recurrent_delta_rule',
    'fused_recurrent_dplr_delta_rule',
    'fused_recurrent_gated_delta_rule',
    'fused_recurrent_gdn',
    'fused_recurrent_gla',
    'fused_recurrent_iplr_delta_rule',
    'fused_recurrent_rwkv6',
    'fused_recurrent_simple_gla',
    'parallel_attn',
    'parallel_simple_gla',
]
