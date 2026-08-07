# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with TRTLLM Attention API."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import FP4Tensor

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    force_use_trtllm_attention,
    supports_trtllm_attention,
    use_trtllm_attention,
)
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
    is_strictly_contiguous,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.flashinfer_trtllm_utils import (
    get_kv_cache_dtype,
    get_kv_cache_shape,
    get_kv_cache_stride_order,
    maybe_quant_query,
    reshape_and_cache_flashinfer,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    get_num_attention_heads_from_layers,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.cp_utils import (
    should_skip_dcp_context_attention,
    split_dcp_context_queries,
)

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

_trtllm_workspace_buffer: torch.Tensor | None = None


def get_trtllm_workspace_buffer() -> torch.Tensor:
    global _trtllm_workspace_buffer
    if _trtllm_workspace_buffer is None:
        _trtllm_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device="cuda",
        )
    return _trtllm_workspace_buffer


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


class TRTLLMBackend(AttentionBackend):
    """Direct TRTLLM attention backend exposed through FlashInfer APIs."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRTLLM"

    @staticmethod
    def get_impl_cls() -> type["TRTLLMImpl"]:
        return TRTLLMImpl

    @staticmethod
    def get_builder_cls() -> type["TRTLLMMetadataBuilder"]:
        return TRTLLMMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return get_kv_cache_stride_order(include_num_layers_dimension)

    @staticmethod
    def get_kv_cache_dtype(kv_cache_dtype: str) -> torch.dtype:
        return get_kv_cache_dtype(kv_cache_dtype)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or vllm_config.model_config is None:
            return [16, 32, 64]

        pc = vllm_config.parallel_config
        mc = vllm_config.model_config
        num_qo_heads = mc.get_num_attention_heads(pc)
        num_kv_heads = mc.get_num_kv_heads(pc)
        if (
            num_kv_heads > 0
            and num_qo_heads // num_kv_heads > 1
            and current_platform.is_device_capability_family(100)
            and can_use_trtllm_attention(num_qo_heads, num_kv_heads)
        ):
            return [16, 32, 64, 128, 256, 512, 1024]
        return [16, 32, 64]

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # trtllm-gen serves both prefill and decode only on SM100/SM103. SM90
        # has the XQA decode kernel but no TRTLLM prefill kernel, so a batch
        # containing any prefill could not be served; keep Hopper and older on
        # the FlashInfer wrapper backend instead.
        return capability.major == 10

    @classmethod
    def _prefill_uses_trtllm(
        cls,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        has_sink: bool,
        vllm_config: VllmConfig,
    ) -> bool:
        """Mirror `use_trtllm_attention`'s prefill decision at selection time.

        `TRTLLMMetadataBuilder.build` has no fallback, so a configuration whose
        prefill slice would not take the TRTLLM path must be rejected here
        rather than raising once traffic arrives.
        """
        # Matches the precedence in `TRTLLMMetadataBuilder.build`: pages >= 128
        # are only served by the trtllm-gen dynamic kernel, so they force the
        # TRTLLM path regardless of the CLI flag.
        if block_size is not None and block_size >= 128:
            return True
        force = vllm_config.attention_config.use_trtllm_attention
        if force is not None:
            return force
        if has_sink:
            return True
        # An unquantized KV cache reaches the builder as "auto", which
        # auto-selects the TRTLLM prefill kernel.
        if kv_cache_dtype is None or not is_quantized_kv_cache(kv_cache_dtype):
            return True
        # With a quantized KV cache the prefill kernel is auto-selected only
        # when the query is quantized to FP8 alongside it.
        return not vllm_config.attention_config.disable_flashinfer_q_quantization

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not (
            supports_trtllm_attention(is_prefill=True)
            and supports_trtllm_attention(is_prefill=False)
        ):
            return "TRTLLM prefill and decode kernels are not both available"

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return None

        model_config = vllm_config.model_config
        if model_config is not None:
            parallel_config = vllm_config.parallel_config
            num_kv_heads = model_config.get_num_kv_heads(parallel_config)
            if num_kv_heads > 0 and not can_use_trtllm_attention(
                model_config.get_num_attention_heads(parallel_config),
                num_kv_heads,
                is_prefill=False,
            ):
                return "TRTLLM decode does not support this query/KV head ratio"

        if not cls._prefill_uses_trtllm(
            kv_cache_dtype, block_size, has_sink, vllm_config
        ):
            return (
                "TRTLLM prefill is not selected for kv_cache_dtype="
                f"{kv_cache_dtype} with the current query quantization settings"
            )

        if vllm_config.parallel_config.decode_context_parallel_size > 1 and has_sink:
            return "TRTLLM DCP with attention sinks is not supported"

        if (
            vllm_config.parallel_config.decode_context_parallel_size > 1
            and device_capability.major == 10
        ):
            return (
                "TRTLLM DCP prefill for SM100 requires an LSE-producing "
                "SM100 new-token prefill path; FMHA-v2 only generates "
                "SM90/SM120 kernels"
            )

        return None

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype == "nvfp4":
            return (
                current_platform.is_device_capability_family(100)
                and supports_trtllm_attention(is_prefill=True)
                and supports_trtllm_attention(is_prefill=False)
            )
        return super().supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256, 512]

    @classmethod
    def supports_sink(cls) -> bool:
        # Respect explicit disable flag (e.g.,
        # --attention-config.use_trtllm_attention=0).
        if force_use_trtllm_attention() is False:
            return False

        if not current_platform.is_device_capability_family(100):
            return False

        return supports_trtllm_attention(
            is_prefill=False
        ) and supports_trtllm_attention(is_prefill=True)

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            return "HND"
        return None

    forward_includes_kv_cache_update: bool = False


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

    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None
    num_dcp_decode_reqs: int = 0
    num_dcp_context_prefill_reqs: int = 0
    num_dcp_decode_tokens: int = 0
    num_dcp_context_prefill_tokens: int = 0


class TRTLLMMetadataBuilder(AttentionMetadataBuilder[TRTLLMMetadata]):
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.attention_config = vllm_config.attention_config

        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.dcp_kv_cache_interleave_size = (
                vllm_config.parallel_config.dcp_kv_cache_interleave_size
            )
        except AssertionError:
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.dcp_kv_cache_interleave_size = 1

        self.num_qo_heads = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or self.model_config.get_num_attention_heads(self.vllm_config.parallel_config)
        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size

        if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
            self.cache_dtype = self.cache_config.cache_dtype
            self.is_kvcache_nvfp4 = self.cache_dtype == "nvfp4"
            if self.is_kvcache_nvfp4:
                if not TRTLLMBackend.supports_kv_cache_dtype("nvfp4"):
                    raise ValueError(
                        "--kv-cache-dtype nvfp4 requires the SM100 trtllm-gen "
                        "FlashInfer path."
                    )
                self.kv_cache_dtype = self.cache_dtype
            else:
                self.kv_cache_dtype = TRTLLMBackend.get_kv_cache_dtype(self.cache_dtype)
        else:
            self.cache_dtype = "auto"
            self.is_kvcache_nvfp4 = False
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        self.q_data_type_prefill = self.get_q_data_type(is_prefill=True)
        self.q_data_type_decode = self.get_q_data_type(is_prefill=False)

        can_use_decode = can_use_trtllm_attention(
            self.num_qo_heads, self.num_kv_heads, is_prefill=False
        )
        assert self.page_size <= 64 or (
            current_platform.is_device_capability_family(100)
            and can_use_decode
            and self.num_qo_heads // self.num_kv_heads > 1
        ), f"Unexpected TRTLLM page size {self.page_size} without trtllm-gen GQA"
        self.use_trtllm_decode_attention = can_use_decode
        self.trtllm_api_decode_kernel: TrtllmDecodeAPIKernel | None = (
            self._get_trtllm_api_decode_kernel() if can_use_decode else None
        )
        supports_spec_as_decode = (
            self.trtllm_api_decode_kernel == TrtllmDecodeAPIKernel.TRTLLM_GEN
        )
        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=supports_spec_as_decode,
            supports_dcp_with_varlen=False,
        )

        per_layer_parameters = get_per_layer_parameters(
            vllm_config, layer_names, TRTLLMImpl
        )
        self.global_hyperparameters = infer_global_hyperparameters(per_layer_parameters)
        self.sm_scale = self.global_hyperparameters.sm_scale
        self.window_left = self.global_hyperparameters.window_left
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap
        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and not TRTLLMBackend.supports_sink():
            raise NotImplementedError(
                "TRTLLM attention sinks require the Blackwell trtllm-gen path."
            )

        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.paged_kv_indptr_gpu = torch.empty(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.dcp_context_kv_lens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )

    def get_q_data_type(self, is_prefill: bool) -> torch.dtype:
        if self.vllm_config.attention_config.disable_flashinfer_q_quantization:
            return self.model_config.dtype

        cache_dtype = self.cache_dtype
        if (
            current_platform.is_device_capability(90)
            and not is_prefill
            and force_use_trtllm_attention() is not False
            and cache_dtype.startswith("fp8")
        ):
            return self.model_config.dtype

        if cache_dtype.startswith("fp8"):
            if current_platform.is_device_capability(
                90
            ) or current_platform.is_device_capability_family(100):
                return TRTLLMBackend.get_kv_cache_dtype(cache_dtype)
            return self.model_config.dtype
        if cache_dtype == "nvfp4":
            return TRTLLMBackend.get_kv_cache_dtype("fp8_e4m3")
        return self.kv_cache_spec.dtype

    @staticmethod
    def _get_trtllm_api_decode_kernel() -> TrtllmDecodeAPIKernel:
        if current_platform.is_device_capability(90):
            return TrtllmDecodeAPIKernel.XQA
        assert current_platform.is_device_capability_family(100)
        return TrtllmDecodeAPIKernel.TRTLLM_GEN

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if current_platform.is_device_capability(90):
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

        if vllm_config.attention_config.use_non_causal:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

        kv_specs = (
            kv_cache_spec.kv_cache_specs.values()
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)
            else [kv_cache_spec]
        )
        num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        for spec in kv_specs:
            if not isinstance(spec, AttentionSpec):
                continue
            if not can_use_trtllm_attention(
                num_qo_heads=num_qo_heads,
                num_kv_heads=spec.num_kv_heads,
                is_prefill=False,
            ):
                return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TRTLLMMetadata:
        if common_prefix_len > 0:
            raise NotImplementedError(
                "TRTLLM attention backend does not support cascade attention."
            )

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        causal = common_attn_metadata.causal
        if not causal:
            raise NotImplementedError(
                "TRTLLM attention backend does not support non-causal attention."
            )

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr = common_attn_metadata.query_start_loc
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu

        uses_spec_reorder = self.reorder_batch_threshold > 1
        prefill_force_trtllm = (
            True
            if self.page_size >= 128
            else self.attention_config.use_trtllm_attention
        )
        prefill_use_trtllm = use_trtllm_attention(
            self.num_qo_heads,
            self.num_kv_heads,
            num_prefill_tokens,
            max_seq_len,
            self.dcp_world_size,
            self.cache_dtype,
            self.q_data_type_prefill,
            is_prefill=True,
            force_use_trtllm=prefill_force_trtllm,
            has_sinks=self.has_sinks,
            has_spec=uses_spec_reorder,
        )
        decode_use_trtllm = self.use_trtllm_decode_attention
        all_uses_trtllm = (num_prefills == 0 or prefill_use_trtllm) and (
            num_decodes == 0 or decode_use_trtllm
        )
        if not all_uses_trtllm:
            raise NotImplementedError(
                "TRTLLM attention backend requires the direct TRTLLM API for "
                "every prefill/decode slice in the batch."
            )

        attn_metadata = TRTLLMMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type_prefill=self.q_data_type_prefill,
            q_data_type_decode=self.q_data_type_decode,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            causal=causal,
            prefill=None,
            decode=None,
        )

        if self.dcp_world_size > 1:
            query_lens = qo_indptr[1:] - qo_indptr[:-1]
            context_kv_lens = seq_lens - query_lens
            local_context_kv_lens = get_dcp_local_seq_lens(
                context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.dcp_kv_cache_interleave_size,
            )
            self.dcp_context_kv_lens[: common_attn_metadata.num_reqs] = (
                local_context_kv_lens
            )
            self.dcp_context_kv_lens[common_attn_metadata.num_reqs :] = 0
            attn_metadata.dcp_context_kv_lens = self.dcp_context_kv_lens[
                : common_attn_metadata.num_reqs
            ]

            skip_dcp_context_attention = False
            if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
                query_lens_cpu = (
                    qo_indptr_cpu[1 : common_attn_metadata.num_reqs + 1]
                    - qo_indptr_cpu[: common_attn_metadata.num_reqs]
                )
                context_kv_lens_cpu = (
                    common_attn_metadata.seq_lens_cpu_upper_bound[
                        : common_attn_metadata.num_reqs
                    ]
                    - query_lens_cpu
                )
                skip_dcp_context_attention = should_skip_dcp_context_attention(
                    context_kv_lens_cpu
                )

            if skip_dcp_context_attention:
                attn_metadata.max_dcp_context_kv_len = 0
            else:
                num_partitions = (
                    self.dcp_world_size * self.dcp_kv_cache_interleave_size
                )
                attn_metadata.max_dcp_context_kv_len = (
                    (max_seq_len + num_partitions - 1) // num_partitions
                ) * self.dcp_kv_cache_interleave_size

            if common_attn_metadata.max_query_len > 1:
                (
                    attn_metadata.num_dcp_decode_reqs,
                    attn_metadata.num_dcp_context_prefill_reqs,
                    attn_metadata.num_dcp_decode_tokens,
                    attn_metadata.num_dcp_context_prefill_tokens,
                ) = split_dcp_context_queries(
                    qo_indptr_cpu,
                    common_attn_metadata.seq_lens_cpu_upper_bound,
                    common_attn_metadata.max_query_len,
                    num_actual_tokens,
                )

        if num_prefills > 0:
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1
            qo_indptr_prefill_gpu = qo_indptr[prefill_start:] - qo_indptr[prefill_start]
            prefill_seq_lens = seq_lens[prefill_start:]
            num_blocks_per_req = (
                prefill_seq_lens + self.page_size - 1
            ) // self.page_size
            paged_kv_indptr_prefill_gpu = self.paged_kv_indptr_gpu[: num_prefills + 1]
            paged_kv_indptr_prefill_gpu[:1] = 0
            torch.cumsum(
                num_blocks_per_req,
                dim=0,
                out=paged_kv_indptr_prefill_gpu[1:],
            )
            query_lens_prefill_cpu = (
                qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
            )
            max_q_len_prefill = int(query_lens_prefill_cpu.max().item())
            attn_metadata.prefill = TRTLLMPrefill(
                block_tables=block_table_tensor[prefill_start:],
                seq_lens=prefill_seq_lens,
                cum_seq_lens_q=qo_indptr_prefill_gpu,
                cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,
                max_q_len=max_q_len_prefill,
                max_seq_len=max_seq_len,
            )

        if num_decodes > 0:
            assert num_decode_tokens % num_decodes == 0, (
                "XQA/trtllm-gen decode requires uniform query lengths per request. "
                f"Got {num_decode_tokens=} and {num_decodes=}."
            )
            assert self.trtllm_api_decode_kernel is not None
            attn_metadata.decode = TRTLLMDecode(
                kernel=self.trtllm_api_decode_kernel,
                block_tables=block_table_tensor[:num_decodes],
                seq_lens=seq_lens[:num_decodes],
                max_seq_len=max_seq_len,
            )

        return attn_metadata


class TRTLLMImpl(AttentionImpl):
    """Implementation for the public TRTLLM attention backend."""

    can_return_lse_for_decode: bool = True
    lse_base_on_e: bool = False
    supports_dcp: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.window_left = (
            self.sliding_window[0] if self.sliding_window is not None else -1
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.is_kvcache_nvfp4 = kv_cache_dtype == "nvfp4"
        self.fp4_data_dim = head_size // 2 if self.is_kvcache_nvfp4 else 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for TRTLLMImpl"
            )

        self.sinks: torch.Tensor | None = None
        self._sinks_source = sinks
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks

        self.supports_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            num_heads, num_kv_heads, is_prefill=False
        )
        vllm_config = get_current_vllm_config_or_none()
        self.supports_quant_query_input = (
            self.supports_xqa_or_trtllm_gen_decode
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and current_platform.is_device_capability_family(100)
            and vllm_config is not None
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        )
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        self.o_sf_scale: float | None = None

        if self.is_kvcache_nvfp4 and vllm_config is not None:
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._nvfp4_fp8_out = torch.empty(
                (max_num_tokens, num_heads, head_size),
                dtype=FP8_DTYPE,
                device="cuda",
            )
        else:
            self._nvfp4_fp8_out = None

        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        if dcp_a2a:
            self.dcp_combine = dcp_a2a_lse_reduce
        else:
            self.dcp_combine = cp_lse_ag_out_rs

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return (
            self.supports_xqa_or_trtllm_gen_decode
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and current_platform.is_device_capability_family(100)
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        source_sinks = self._sinks_source
        if source_sinks is None:
            return
        if source_sinks.dtype == torch.float32:
            self.sinks = source_sinks
        elif self.sinks is None or self.sinks.dtype != torch.float32:
            self.sinks = source_sinks.to(torch.float32)
        else:
            self.sinks.copy_(source_sinks)

    def get_xqa_bmm1_scale(self, layer: torch.nn.Module, q_data_type: torch.dtype):
        bmm1_scale = self.scale
        if is_quantized_kv_cache(self.kv_cache_dtype):
            if q_data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
                bmm1_scale *= layer._q_scale_float
            bmm1_scale *= layer._k_scale_float
        return bmm1_scale

    def maybe_quant_query(
        self,
        query: torch.Tensor,
        q_data_type: torch.dtype,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return maybe_quant_query(query, q_data_type, scale)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TRTLLMMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._v_scale_float

        prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)
        decode_kernel = (
            attn_metadata.decode.kernel
            if isinstance(attn_metadata.decode, TRTLLMDecode)
            else None
        )
        decode_with_xqa = decode_kernel == TrtllmDecodeAPIKernel.XQA
        decode_with_trtllm_gen = decode_kernel == TrtllmDecodeAPIKernel.TRTLLM_GEN
        decode_use_trtllm = decode_with_xqa or decode_with_trtllm_gen

        if output_scale is None:
            assert output_block_scale is None, (
                "output_block_scale is not supported when fusion has not happened"
            )
        else:
            assert attn_metadata.q_data_type_prefill == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened for prefill."
            )
            assert attn_metadata.q_data_type_decode == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened for decode."
            )
            assert (attn_metadata.num_prefills == 0 or prefill_use_trtllm) and (
                attn_metadata.num_decodes == 0 or decode_with_trtllm_gen
            ), "Output quant fusion requires TRTLLM prefill/trtllm-gen decode"

            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None, (
                    "output_block_scale should not be provided for fp8 output"
                )
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None, (
                    "output_block_scale is required for nvfp4 output"
                )
            else:
                raise ValueError(f"Unsupported output dtype: {output.dtype}")

            if layer._o_scale_float is None:
                layer._o_scale_float = output_scale.cpu().item()
                if output.dtype == FP8_DTYPE:
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float
                elif output.dtype == FP4_DTYPE:
                    self.o_sf_scale = layer._o_scale_float

        num_actual_tokens = attn_metadata.num_actual_tokens

        if not self.is_kvcache_nvfp4 and kv_cache.dtype == torch.uint8:
            fp8_view_dtype = None
            if self.kv_cache_dtype in ("fp8", "fp8_e4m3", torch.float8_e4m3fn):
                fp8_view_dtype = torch.float8_e4m3fn
            elif self.kv_cache_dtype in ("fp8_e5m2", torch.float8_e5m2):
                fp8_view_dtype = torch.float8_e5m2
            if fp8_view_dtype is not None:
                kv_cache = kv_cache.view(fp8_view_dtype)

        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        stride_order = TRTLLMBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)
        kv_cache_permute = canonicalize_singleton_dim_strides(kv_cache_permute)

        hs = self.head_size
        nvfp4_kv_data = None
        nvfp4_kv_block_scales = None
        if self.is_kvcache_nvfp4:
            k_cache, v_cache = kv_cache.split(self.num_kv_heads, dim=1)
            kv_cache_tuple = (
                canonicalize_singleton_dim_strides(k_cache.permute(*stride_order)),
                canonicalize_singleton_dim_strides(v_cache.permute(*stride_order)),
            )
            k_data, k_sf = nvfp4_split_data_scale(kv_cache_tuple[0])
            v_data, v_sf = nvfp4_split_data_scale(kv_cache_tuple[1])
            nvfp4_kv_data = (k_data, v_data)
            nvfp4_kv_block_scales = (k_sf, v_sf)
        else:
            kv_cache_tuple = kv_cache_permute.split(hs, dim=-1)

        assert prefill_use_trtllm or decode_use_trtllm, (
            "TRTLLMImpl only supports direct TRTLLM prefill/decode metadata"
        )
        if num_prefill_tokens > 0:
            assert prefill_use_trtllm
            self.forward_prefill_trtllm(
                layer=layer,
                query=query,
                kv_cache_permute=kv_cache_permute,
                kv_cache_tuple=kv_cache_tuple,
                nvfp4_kv_data=nvfp4_kv_data,
                nvfp4_kv_block_scales=nvfp4_kv_block_scales,
                attn_metadata=attn_metadata,
                output=output,
                output_block_scale=output_block_scale,
                num_decode_tokens=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens,
            )
        if num_decode_tokens > 0:
            assert decode_use_trtllm
            self.forward_decode_trtllm(
                layer=layer,
                query=query,
                kv_cache_permute=kv_cache_permute,
                kv_cache_tuple=kv_cache_tuple,
                nvfp4_kv_data=nvfp4_kv_data,
                nvfp4_kv_block_scales=nvfp4_kv_block_scales,
                attn_metadata=attn_metadata,
                output=output,
                output_block_scale=output_block_scale,
                num_decode_tokens=num_decode_tokens,
                decode_with_xqa=decode_with_xqa,
                decode_with_trtllm_gen=decode_with_trtllm_gen,
            )

        return output_padded

    def forward_prefill_trtllm(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache_permute: torch.Tensor,
        kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
        nvfp4_kv_data: torch.Tensor | None,
        nvfp4_kv_block_scales: torch.Tensor | None,
        attn_metadata: TRTLLMMetadata,
        output: torch.Tensor,
        output_block_scale: torch.Tensor | None,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> None:
        prefill_query = query[num_decode_tokens:]
        assert prefill_query.shape[0] == num_prefill_tokens

        prefill_query = self.maybe_quant_query(
            query=prefill_query,
            q_data_type=attn_metadata.q_data_type_prefill,
            scale=layer._q_scale,
        )

        assert isinstance(attn_metadata.prefill, TRTLLMPrefill)
        prefill_query = canonicalize_singleton_dim_strides(prefill_query.contiguous())
        workspace_buffer = get_trtllm_workspace_buffer()
        block_tables_prefill = attn_metadata.prefill.block_tables
        seq_lens_prefill = attn_metadata.prefill.seq_lens

        assert get_kv_cache_layout() == "HND"
        assert is_strictly_contiguous(prefill_query)
        assert is_strictly_contiguous(workspace_buffer)
        assert is_strictly_contiguous(block_tables_prefill)
        assert is_strictly_contiguous(seq_lens_prefill)

        if output.dtype == FP4_DTYPE:
            if self.dcp_world_size > 1:
                raise NotImplementedError(
                    "TRTLLM DCP does not support fused NVFP4 attention output."
                )
            assert self.o_sf_scale is not None
            out = FP4Tensor(
                data=output[num_decode_tokens:],
                scale=output_block_scale,
                scale_start_index=num_decode_tokens,
                original_shape=prefill_query.shape,
            )
        else:
            assert self.o_sf_scale is None
            out = output[num_decode_tokens:]

        needs_fp8_out = (
            self.dcp_world_size <= 1
            and self.is_kvcache_nvfp4
            and output.dtype != FP8_DTYPE
        )
        if needs_fp8_out:
            out = self._nvfp4_fp8_out[:num_prefill_tokens]

        if self.dcp_world_size > 1:
            self.forward_prefill_trtllm_dcp(
                layer=layer,
                prefill_query=prefill_query,
                key=key,
                value=value,
                kv_cache_tuple=kv_cache_tuple,
                nvfp4_kv_data=nvfp4_kv_data,
                nvfp4_kv_block_scales=nvfp4_kv_block_scales,
                attn_metadata=attn_metadata,
                out=out,
                num_decode_tokens=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens,
            )
            return

        prefill_kv_block_scales = None
        if self.is_kvcache_nvfp4:
            assert attn_metadata.q_data_type_prefill == FP8_DTYPE, (
                "NVFP4 KV cache requires FP8 quantized queries for "
                "trtllm-gen prefill. Set disable_flashinfer_q_quantization=False."
            )
            mock_kv_cache = nvfp4_kv_data
            mock_block_table = block_tables_prefill
            prefill_kv_block_scales = nvfp4_kv_block_scales
        elif (
            attn_metadata.q_data_type_prefill != FP8_DTYPE
            and self.kv_cache_dtype.startswith("fp8")
        ):
            kv_cache_permute = canonicalize_singleton_dim_strides(kv_cache_permute)
            kv_strides = kv_cache_permute.stride()
            assert (
                kv_strides[-1] == 1 and kv_strides[-2] == kv_cache_permute.shape[-1]
            ), (
                "KV cache inner dims (block_size, head_size) must be "
                f"contiguous, got strides {kv_strides}"
            )
            hs = self.head_size
            b_kv, h_kv, n_kv = kv_cache_permute.shape[:3]
            kv_cache_5d = kv_cache_permute.view(b_kv, h_kv, n_kv, 2, hs)
            kv_cache_5d = kv_cache_5d.permute(0, 3, 1, 2, 4)
            mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                kv_cache_5d,
                block_tables_prefill,
                layer._k_scale,
                layer._v_scale,
                attn_metadata.q_data_type_prefill,
            )
        else:
            mock_kv_cache = kv_cache_tuple
            mock_block_table = block_tables_prefill

        trtllm_batch_context_with_kv_cache(
            query=prefill_query,
            kv_cache=mock_kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=mock_block_table,
            seq_lens=seq_lens_prefill,
            max_q_len=attn_metadata.prefill.max_q_len,
            max_kv_len=attn_metadata.prefill.max_seq_len,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            batch_size=attn_metadata.num_prefills,
            cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
            cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
            window_left=self.window_left,
            sinks=self.sinks,
            o_sf_scale=self.o_sf_scale,
            out=out,
            kv_cache_sf=prefill_kv_block_scales,
        )

        if needs_fp8_out:
            output[num_decode_tokens : num_decode_tokens + num_prefill_tokens].copy_(
                out[:num_prefill_tokens].to(output.dtype)
            )

    def forward_prefill_trtllm_dcp(
        self,
        layer: torch.nn.Module,
        prefill_query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
        nvfp4_kv_data: torch.Tensor | None,
        nvfp4_kv_block_scales: torch.Tensor | None,
        attn_metadata: TRTLLMMetadata,
        out: torch.Tensor,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> None:
        assert isinstance(attn_metadata.prefill, TRTLLMPrefill)
        assert attn_metadata.dcp_context_kv_lens is not None
        assert attn_metadata.max_dcp_context_kv_len is not None
        assert self.sinks is None, "TRTLLM DCP does not support sinks"
        raise NotImplementedError(
            "TRTLLM DCP prefill requires an LSE-producing SM100 new-token "
            "prefill path. The old FMHA-v2 suffix path is intentionally not "
            "used because it only generates SM90/SM120 kernels."
        )

    def forward_decode_trtllm(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache_permute: torch.Tensor,
        kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
        nvfp4_kv_data: torch.Tensor | None,
        nvfp4_kv_block_scales: torch.Tensor | None,
        attn_metadata: TRTLLMMetadata,
        output: torch.Tensor,
        output_block_scale: torch.Tensor | None,
        num_decode_tokens: int,
        decode_with_xqa: bool,
        decode_with_trtllm_gen: bool,
    ) -> None:
        decode_query = query[:num_decode_tokens]
        assert decode_query.shape[0] == num_decode_tokens

        decode_query = self.maybe_quant_query(
            query=decode_query,
            q_data_type=attn_metadata.q_data_type_decode,
            scale=layer._q_scale,
        )

        assert isinstance(attn_metadata.decode, TRTLLMDecode)
        decode_query = canonicalize_singleton_dim_strides(decode_query.contiguous())
        workspace_buffer = get_trtllm_workspace_buffer()
        block_tables_decode = attn_metadata.decode.block_tables
        seq_lens_decode = attn_metadata.decode.seq_lens

        if decode_with_trtllm_gen:
            assert get_kv_cache_layout() == "HND"
        else:
            assert decode_with_xqa
        assert is_strictly_contiguous(decode_query)
        assert is_strictly_contiguous(workspace_buffer)
        assert is_strictly_contiguous(block_tables_decode)
        assert is_strictly_contiguous(seq_lens_decode)
        kv_cache_permute = canonicalize_singleton_dim_strides(kv_cache_permute)
        kv_strides = kv_cache_permute.stride()
        assert kv_strides[-1] == 1 and kv_strides[-2] == kv_cache_permute.shape[-1], (
            "KV cache inner dims (block_size, head_size) must be "
            f"contiguous, got strides {kv_strides}"
        )

        if output.dtype == FP4_DTYPE:
            if self.dcp_world_size > 1:
                raise NotImplementedError(
                    "TRTLLM DCP does not support fused NVFP4 attention output."
                )
            assert self.o_sf_scale is not None
            out = FP4Tensor(
                data=output[:num_decode_tokens],
                scale=output_block_scale,
                scale_start_index=0,
                original_shape=decode_query.shape,
            )
        else:
            assert self.o_sf_scale is None
            out = output[:num_decode_tokens]

        needs_fp8_out = (
            self.dcp_world_size <= 1
            and self.is_kvcache_nvfp4
            and output.dtype != FP8_DTYPE
        )
        if needs_fp8_out:
            out = self._nvfp4_fp8_out[:num_decode_tokens]

        if num_decode_tokens % attn_metadata.num_decodes != 0:
            q_len_per_req = 1
        else:
            q_len_per_req = num_decode_tokens // attn_metadata.num_decodes

        if decode_with_xqa and q_len_per_req > 1:
            raise NotImplementedError(
                "FlashInfer XQA speculative decode is not wired in vLLM yet."
            )

        bmm1_scale = (
            self.get_xqa_bmm1_scale(layer, attn_metadata.q_data_type_decode)
            if decode_with_xqa
            else self.bmm1_scale
        )

        if self.dcp_world_size > 1:
            assert decode_with_trtllm_gen
            assert attn_metadata.dcp_context_kv_lens is not None
            assert attn_metadata.max_dcp_context_kv_len is not None
            assert self.sinks is None, "TRTLLM DCP does not support sinks"
            decode_query = get_dcp_group().all_gather(
                decode_query.contiguous(), dim=1
            )
            dcp_out = torch.empty(
                decode_query.shape,
                dtype=out.dtype,
                device=decode_query.device,
            )
            dcp_lse = torch.empty(
                (decode_query.shape[0], decode_query.shape[1]),
                dtype=torch.float32,
                device=decode_query.device,
            )
            trtllm_batch_decode_with_kv_cache(
                query=decode_query,
                kv_cache=nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_tuple,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables_decode,
                seq_lens=attn_metadata.dcp_context_kv_lens[
                    : attn_metadata.num_decodes
                ],
                max_seq_len=attn_metadata.max_dcp_context_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=self.bmm2_scale,
                window_left=self.window_left,
                o_sf_scale=self.o_sf_scale,
                out=dcp_out,
                kv_layout=get_kv_cache_layout(),
                backend=attn_metadata.decode.kernel.value,
                q_len_per_req=q_len_per_req,
                kv_cache_sf=nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None,
                lse=dcp_lse,
                return_lse=True,
            )
            out[:num_decode_tokens] = self.dcp_combine(
                dcp_out,
                dcp_lse,
                get_dcp_group(),
                is_lse_base_on_e=False,
            )
            if needs_fp8_out:
                output[:num_decode_tokens].copy_(out.to(output.dtype))
            return

        trtllm_batch_decode_with_kv_cache(
            query=decode_query,
            kv_cache=nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_tuple,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables_decode,
            seq_lens=seq_lens_decode,
            max_seq_len=attn_metadata.decode.max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            window_left=self.window_left,
            sinks=self.sinks,
            o_sf_scale=self.o_sf_scale,
            out=out,
            kv_layout=get_kv_cache_layout(),
            backend=attn_metadata.decode.kernel.value,
            q_len_per_req=q_len_per_req,
            kv_cache_sf=nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None,
        )

        if needs_fp8_out:
            output[:num_decode_tokens].copy_(out.to(output.dtype))

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.kv_sharing_target_layer_name is None:
            reshape_and_cache_flashinfer(
                key,
                value,
                kv_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
                self.num_kv_heads,
                self.head_size,
                self.is_kvcache_nvfp4,
            )
