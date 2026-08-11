# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for FlashInfer wrapper and TRTLLM attention backends."""

from dataclasses import dataclass
from enum import Enum

import torch

from vllm import _custom_ops as custom_ops
from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim
from vllm.v1.attention.backends.utils import get_kv_cache_layout

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

_trtllm_workspace_buffer: torch.Tensor | None = None


class TrtllmDecodeAPIKernel(Enum):
    """Decode kernels selected inside the TRTLLM backend."""

    XQA = "xqa"
    TRTLLM_GEN = "trtllm-gen"


@dataclass
class TRTLLMPrefill:
    """Metadata for the TRTLLM prefill pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to prefill requests.
    Shape: [num_prefills, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to prefill requests.
    Shape: [num_prefills]
    """

    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor

    max_q_len: int
    """
    The maximum query length *among prefill requests*.
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


@dataclass
class TRTLLMDecode:
    """Metadata for decode paths using FlashInfer's TRTLLM decode API.

    FlashInfer exposes both XQA (SM90) and trtllm-gen (SM100) through
    ``trtllm_batch_decode_with_kv_cache``. Keep them as distinct vLLM
    decode kernels because their dtype/layout/output constraints differ.
    """

    kernel: TrtllmDecodeAPIKernel

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to decode requests.
    Shape: [num_decodes, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to decode requests.
    Shape: [num_decodes]
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


@dataclass
class TRTLLMMetadata:
    num_actual_tokens: int
    """Total number of tokens in the batch (excluding padding)."""

    slot_mapping: torch.Tensor
    """Tensor for writing K/V to the cache. Shape: [num_actual_tokens]"""

    q_data_type_prefill: torch.dtype
    q_data_type_decode: torch.dtype

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    causal: bool

    prefill: TRTLLMPrefill | None
    """Direct TRTLLM API metadata for the prefill slice."""

    decode: TRTLLMDecode | None
    """Direct TRTLLM API metadata for the decode slice."""


def get_trtllm_workspace_buffer() -> torch.Tensor:
    global _trtllm_workspace_buffer
    if _trtllm_workspace_buffer is None:
        _trtllm_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device="cuda",
        )
    return _trtllm_workspace_buffer


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


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    src_stride_page,
    src_stride_kv,
    src_stride_head,
    src_stride_block,
    src_stride_head_size,
    DST_K_CACHE_STRIDE: tl.constexpr,
    DST_KV_CACHE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    mock_page_idx = batch_idx * block_table_stride + mock_block_table_idx + 1
    logical_offsets = tl.arange(0, HEAD_STRIDE)
    block_offsets = logical_offsets // HEAD_SIZE
    head_size_offsets = logical_offsets % HEAD_SIZE

    for h in range(NUM_KV_HEADS):
        h_off = tl.cast(h, tl.int64)

        src_k = (
            orig_page_num * src_stride_page
            + h_off * src_stride_head
            + block_offsets * src_stride_block
            + head_size_offsets * src_stride_head_size
        )
        fp8_k = tl.load(kv_cache_ptr + src_k)
        dequant_k = (fp8_k.to(tl.float32) * k_scale_val).to(dequant_dtype)

        dst_k = mock_page_idx * DST_KV_CACHE_STRIDE + h * HEAD_STRIDE + logical_offsets
        tl.store(mock_kv_cache_ptr + dst_k, dequant_k)

        src_v = (
            orig_page_num * src_stride_page
            + src_stride_kv
            + h_off * src_stride_head
            + block_offsets * src_stride_block
            + head_size_offsets * src_stride_head_size
        )
        fp8_v = tl.load(kv_cache_ptr + src_v)
        dequant_v = (fp8_v.to(tl.float32) * v_scale_val).to(dequant_dtype)

        dst_v = (
            mock_page_idx * DST_KV_CACHE_STRIDE
            + DST_K_CACHE_STRIDE
            + h * HEAD_STRIDE
            + logical_offsets
        )
        tl.store(mock_kv_cache_ptr + dst_v, dequant_v)


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    strides = kv_cache.stride()
    num_kv_heads, block_size, head_size = s[2], s[3], s[4]
    head_stride = block_size * head_size
    k_cache_stride = num_kv_heads * head_stride
    kv_cache_stride = k_cache_stride * s[1]

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        k_cache_stride,
        kv_cache_stride,
        head_stride,
        head_size,
        num_kv_heads,
    )
    return mock_kv_cache, mock_block_table


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
