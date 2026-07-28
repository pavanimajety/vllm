# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct TRTLLM attention backend exposed through FlashInfer APIs."""

from typing import ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    force_use_trtllm_attention,
    supports_trtllm_attention,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.attention.backends.flashinfer import (
    FlashInferImpl,
    FlashInferMetadataBuilder,
    FlashInferOrTRTLLMMetadata,
    TRTLLMDecode,
    TrtllmDecodeAPIKernel,
    TRTLLMMetadata,
    TRTLLMPrefill,
)
from vllm.v1.attention.backends.flashinfer_trtllm_utils import (
    get_kv_cache_dtype,
    get_kv_cache_shape,
    get_kv_cache_stride_order,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType
from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs


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
        return capability >= DeviceCapability(8, 0) and capability <= DeviceCapability(
            12, 1
        )

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


class TRTLLMMetadataBuilder(AttentionMetadataBuilder[TRTLLMMetadata]):
    metadata_cls: ClassVar[type[TRTLLMMetadata]] = TRTLLMMetadata
    use_direct_trtllm_api: ClassVar[bool] = True

    __init__ = FlashInferMetadataBuilder.__init__
    _make_buffer = FlashInferMetadataBuilder._make_buffer
    get_q_data_type = FlashInferMetadataBuilder.get_q_data_type
    _get_workspace_buffer = FlashInferMetadataBuilder._get_workspace_buffer
    set_workspace_buffer = FlashInferMetadataBuilder.set_workspace_buffer
    _get_prefill_wrapper = FlashInferMetadataBuilder._get_prefill_wrapper
    _get_decode_wrapper = FlashInferMetadataBuilder._get_decode_wrapper
    _get_trtllm_api_decode_kernel = (
        FlashInferMetadataBuilder._get_trtllm_api_decode_kernel
    )
    _compute_flashinfer_kv_metadata = (
        FlashInferMetadataBuilder._compute_flashinfer_kv_metadata
    )
    build = FlashInferMetadataBuilder.build

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


class TRTLLMImpl(AttentionImpl):
    """Implementation for the public TRTLLM attention backend."""

    can_return_lse_for_decode: bool = True

    __init__ = FlashInferImpl.__init__
    fused_output_quant_supported = FlashInferImpl.fused_output_quant_supported
    process_weights_after_loading = FlashInferImpl.process_weights_after_loading
    get_xqa_bmm1_scale = FlashInferImpl.get_xqa_bmm1_scale
    maybe_quant_query = FlashInferImpl.maybe_quant_query
    forward_with_trtllm_api = FlashInferImpl.forward_with_trtllm_api
    forward_prefill_trtllm = FlashInferImpl.forward_prefill_trtllm
    forward_decode_trtllm = FlashInferImpl.forward_decode_trtllm
    do_kv_cache_update = FlashInferImpl.do_kv_cache_update

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferOrTRTLLMMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, TRTLLMMetadata)
        return FlashInferImpl.forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )


__all__ = [
    "TRTLLMBackend",
    "TRTLLMDecode",
    "TRTLLMImpl",
    "TRTLLMMetadata",
    "TRTLLMMetadataBuilder",
    "TRTLLMPrefill",
    "TrtllmDecodeAPIKernel",
]
