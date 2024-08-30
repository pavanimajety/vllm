from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp8_linear, create_per_tensor_scale_param, cutlass_fp8_supported,
    requantize_with_max_scale)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

ACTIVATION_SCHEMES = ["static"]


class ModelOptFp8Config(QuantizationConfig):
    """Config class for ModelOpt FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "static",
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme

    @classmethod
    def get_name(cls) -> str:
        return "modelopt"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp8Config":
        quant_config = cls.get_from_keys(config, ["quantization"])
        quant_method = quant_config["quant_algo"]
        is_checkpoint_fp8_serialized = ("FP8" in quant_method)
        activation_scheme = "static"
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import
        if isinstance(layer, LinearBase):
            return ModelOptFp8LinearMethod(self)
        elif isinstance(layer, Attention):
            return ModelOptFp8KVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class ModelOptFp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        super().__init__(quant_config)


class ModelOptFp8LinearMethod(LinearMethodBase):
    """Linear method for Model Optimizer static quantization.
    Supports loading FP8 checkpoints with static weight scale and
    activation scale. Future support might be added for dynamic 
    scales.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn datatype 
        Args: quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.process_after_load = True
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(
            weight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        # If checkpoint is fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            weight_scale = create_per_tensor_scale_param(
                output_partition_sizes, **extra_weight_attrs)
            layer.register_parameter("weight_scale", weight_scale)
            input_scale = create_per_tensor_scale_param(
                output_partition_sizes, **extra_weight_attrs)
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return
        # If checkpoint is fp/bf16 and not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.logical_widths = None
            layer.input_scale = None
            return

        else:
            # weight_dtype = torch.float8_e4m3fn
            weight_scale = layer.weight_scale.to(torch.float32)
            max_w_scale, weight = requantize_with_max_scale(
                layer.weight, layer.weight_scale, layer.logical_widths)
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
            layer.input_scale = Parameter(layer.input_scale.max(),
                                          requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported)
