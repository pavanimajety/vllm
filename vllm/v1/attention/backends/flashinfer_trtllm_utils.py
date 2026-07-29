# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for FlashInfer wrapper and TRTLLM attention backends."""

import torch

from vllm import _custom_ops as custom_ops
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim
from vllm.v1.attention.backends.utils import get_kv_cache_layout


def get_kv_cache_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    if kv_cache_dtype == "nvfp4":
        return torch.uint8
    raise ValueError(f"Unrecognized dtype: {kv_cache_dtype}")


def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "auto",
) -> tuple[int, ...]:
    if cache_dtype_str == "nvfp4":
        full_dim = nvfp4_kv_cache_full_dim(head_size)
        return (num_blocks, 2 * num_kv_heads, block_size, full_dim)
    return (num_blocks, num_kv_heads, block_size, 2 * head_size)


def get_kv_cache_stride_order(
    include_num_layers_dimension: bool = False,
) -> tuple[int, ...]:
    cache_layout = get_kv_cache_layout()
    if cache_layout == "NHD" and include_num_layers_dimension:
        return (1, 0, 3, 2, 4)
    if cache_layout == "NHD":
        return (0, 2, 1, 3)
    if cache_layout == "HND" and include_num_layers_dimension:
        return (1, 2, 0, 3, 4)
    if cache_layout == "HND":
        return (0, 1, 2, 3)
    raise ValueError(f"Unknown cache layout format {cache_layout}.")


def maybe_quant_query(
    query: torch.Tensor,
    q_data_type: torch.dtype,
    scale: torch.Tensor,
) -> torch.Tensor:
    if query.dtype == q_data_type:
        return query
    assert query.dtype in [torch.float16, torch.bfloat16]
    assert q_data_type in [torch.float8_e4m3fn, torch.float8_e5m2]
    assert query.dim() == 3
    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]
    assert query.stride(2) == 1 and query.stride(1) == head_size
    query_quantized, _ = custom_ops.scaled_fp8_quant(
        query.view(num_tokens, num_heads * head_size), scale=scale
    )
    return query_quantized.view(num_tokens, num_heads, head_size)


def reshape_and_cache_flashinfer(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    is_kvcache_nvfp4: bool,
) -> None:
    if is_kvcache_nvfp4:
        k_cache, v_cache = kv_cache.transpose(1, 2).split(num_kv_heads, dim=-2)
    else:
        k_cache, v_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
