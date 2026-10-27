# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for miscellaneous utilities
"""

from typing import Optional

import pytest
import torch

from tests.ops.rotary_embedding_op import RotaryEmbedding
from tests.utils import opcheck


def rotary_embedding_opcheck(rot,
                             positions: torch.Tensor,
                             query: torch.Tensor,
                             key: Optional[torch.Tensor] = None,
                             offsets: Optional[torch.Tensor] = None):
    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)

    # ops.rotary_embedding()/batched_rotary_embedding()
    # are in-place operations that update the query and key tensors.
    if offsets is not None:
        raise NotImplementedError(
            "batched_rotary_embedding is not implemented yet.")
    else:
        opcheck(torch.ops._C.rotary_embedding,
                (positions, query, key, rot.head_size, cos_sin_cache,
                 rot.is_neox_style))


#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "max_position": [11],
        "head_size": [32],
        "seq_len": [11],
    },
}


@pytest.mark.parametrize("device", ["xpu"])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim", [32])
@pytest.mark.parametrize("head_size", [32, 108])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("head_stride_is_contiguous", [True, False])
def test_rotary_embedding_opcheck(device, max_position, is_neox_style,
                                  rotary_dim, head_size, seq_len, use_key,
                                  head_stride_is_contiguous):
    batch_size = 1
    base = 10000
    num_heads = 7
    rot = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, torch.float32)

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device=device)
    head_stride = head_size + (64 if head_stride_is_contiguous else 0)

    query = torch.randn(batch_size,
                        seq_len,
                        num_heads,
                        head_stride,
                        dtype=torch.float32,
                        device=device)
    key = torch.randn_like(query) if use_key else None
    query = query[..., :head_size]
    key = key[..., :head_size] if use_key else None

    rotary_embedding_opcheck(rot, positions, query, key)

    # if we have a contiguous head stride, test the alternate
    # [..., num_heads * head_dim] shape/layout
    if head_stride_is_contiguous:
        rotary_embedding_opcheck(
            rot, positions, query.flatten(start_dim=-2),
            key.flatten(start_dim=-2) if use_key else None)


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rope_dim_offset", [0, 16, 64])
@pytest.mark.parametrize("use_key", [True, False])
def test_rotary_embedding_rope_dim_offset_and_inverse(is_neox_style,
                                                      rope_dim_offset,
                                                      use_key):
    device = "xpu"
    num_tokens, num_heads, head_size, rotary_dim = 17, 4, 128, 64
    max_position = 4096
    rot = RotaryEmbedding(head_size, rotary_dim, max_position, 10000,
                          is_neox_style, torch.float32)
    cos_sin_cache = rot.cos_sin_cache.to(device, dtype=torch.float32)
    positions = torch.randint(0, max_position, (num_tokens, ), device=device)

    query = torch.randn(num_tokens,
                        num_heads,
                        head_size,
                        dtype=torch.float32,
                        device=device)
    key = torch.randn_like(query) if use_key else None
    rot_dims = slice(rope_dim_offset, rope_dim_offset + rotary_dim)

    # Reference: rotate only the rope slice of each head, with no offset.
    ref_query = query.clone()
    ref_key = key.clone() if use_key else None
    torch.ops._C.rotary_embedding(positions, ref_query[..., rot_dims],
                                  ref_key[..., rot_dims] if use_key else None,
                                  rotary_dim, cos_sin_cache, is_neox_style)

    out_query = query.clone()
    out_key = key.clone() if use_key else None
    torch.ops._C.rotary_embedding(positions, out_query, out_key, head_size,
                                  cos_sin_cache, is_neox_style,
                                  rope_dim_offset, False)
    torch.testing.assert_close(out_query, ref_query)
    if use_key:
        torch.testing.assert_close(out_key, ref_key)

    # The inverse rotation undoes the forward rotation.
    torch.ops._C.rotary_embedding(positions, out_query, out_key, head_size,
                                  cos_sin_cache, is_neox_style,
                                  rope_dim_offset, True)
    torch.testing.assert_close(out_query, query, atol=1e-5, rtol=1e-5)
    if use_key:
        torch.testing.assert_close(out_key, key, atol=1e-5, rtol=1e-5)

    opcheck(torch.ops._C.rotary_embedding,
            (positions, query, key, head_size, cos_sin_cache, is_neox_style,
             rope_dim_offset, True))


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rotary_embedding_cache_dtype_differs_from_query(
        is_neox_style, dtype):
    # vLLM may keep the cos/sin cache in fp32 for reduced-precision query/key.
    device = "xpu"
    num_tokens, num_heads, head_size, rotary_dim = 17, 4, 64, 64
    max_position = 4096
    rot = RotaryEmbedding(head_size, rotary_dim, max_position, 10000,
                          is_neox_style, torch.float32)
    cos_sin_cache = rot.cos_sin_cache.to(device, dtype=torch.float32)
    positions = torch.randint(0, max_position, (num_tokens, ), device=device)
    query = torch.randn(num_tokens,
                        num_heads,
                        head_size,
                        dtype=torch.float32,
                        device=device)

    ref_query = query.to(dtype).float()
    torch.ops._C.rotary_embedding(positions, ref_query, None, head_size,
                                  cos_sin_cache, is_neox_style)

    out_query = query.to(dtype)
    torch.ops._C.rotary_embedding(positions, out_query, None, head_size,
                                  cos_sin_cache, is_neox_style)

    torch.testing.assert_close(out_query, ref_query.to(dtype))
