from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.fp8 import (Fp8Config,
                                                         cutlass_fp8_supported,
                                                         per_tensor_quantize)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class ModelOptQuantizer(torch.nn.Module):
    """Class to load amax values for Model Opt checkpoints."""

    def __init__(self, _amax, **extra_weight_attrs):
        super().__init__()
        self._amax = _amax
        set_weight_attrs(
            _amax,
            {
                **extra_weight_attrs,
                "needs_scalar_to_array": True,
            },
        )
        return

    def forward(self, x):
        return x


class ModelOptFp8LinearMethod(LinearMethodBase):
    """Linear method for Model Optimizer static quantization.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Limitations[Same as Fp8LinearMethod]:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
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
        # Model Opt weights are not converted to FP8 when stored in
        # the checkpoint, so we use the original datatype. May change
        # in the future if the format of Model Opt checkpoint changes.
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
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

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            weight_amax = Parameter(
                torch.empty(len(output_partition_sizes), dtype=torch.float32),
                requires_grad=False,
            )

            layer.add_module(
                "weight_quantizer",
                ModelOptQuantizer(weight_amax, **extra_weight_attrs),
            )

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                input_amax = Parameter(
                    torch.empty(len(output_partition_sizes),
                                dtype=torch.float32),
                    requires_grad=False,
                )
                layer.add_module(
                    "input_quantizer",
                    ModelOptQuantizer(input_amax, **extra_weight_attrs),
                )

    def scales_shard_indexer(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: Union[str, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv_idxs = {"q": 0, "k": 1, "v": 2}

        if isinstance(shard_id, int):
            pass
        elif isinstance(shard_id, str):
            if shard_id not in qkv_idxs:
                raise ValueError(f"Unknown shard_id: {shard_id}")
            shard_id = qkv_idxs[shard_id]
        else:
            ValueError(f"Shard id must be int or str but got {type(shard_id)}")

        return param[shard_id], loaded_weight

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

        # If checkpoint is fp8, requantize the separately quantized logical
        # weights into a single fp8 weight with a single weight scale.
        else:
            # WEIGHT_SCALE / WEIGHT
            #   Loop over logical weights, requantizing with single scale.
            # Convert the given weight to fp8 because model opt generates
            # quantization scales, but doesn't convert then to fp8.
            layer.weight_scale = layer.weight_quantizer._amax / 448
            max_w_scale = layer.weight_scale.max()

            weight = torch.empty_like(layer.weight, dtype=torch.float8_e4m3fn)
            start = 0
            for idx, logical_width in enumerate(layer.logical_widths):
                end = start + logical_width
                weight_dq = layer.weight[start:end, :]
                weight[start:end, :] = per_tensor_quantize(
                    weight_dq, layer.weight_scale.max())
                start = end

            layer.weight.data = weight
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

            # WEIGHT
            #   Transpose weight for passing to torch._scaled_mm
            layer.weight = Parameter(weight.t(), requires_grad=False)

            # INPUT ACTIVATION SCALE
            #   Dynamic: set to None (required input to ops.scaled_fp8_quant).
            #   Static:  set to max of the input_scales (since they are equal).
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = layer.input_quantizer._amax
                if not all_close_1d(layer.input_scale):
                    raise ValueError(
                        "All the input_scales for the logical weights of a "
                        f"layer must be equal. But got {layer.input_scale}")
                layer.input_scale = Parameter(layer.input_scale.max(),
                                              requires_grad=False)
            else:
                raise ValueError(
                    f"Unsupported scheme {self.quant_config.activation_scheme} "
                    "for Model Optimizer weights")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_scale computed from x.
        #   If static, layer.input_scale is scalar and x_scale is input_scale.

        if bias is None and self.cutlass_fp8_supported:
            qinput, x_scale = ops.scaled_fp8_quant(x, layer.input_scale)

            # Fused GEMM_DQ
            output = ops.cutlass_scaled_mm(
                qinput,
                layer.weight,
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
            )

        else:
            qinput, x_scale = ops.scaled_fp8_quant(x,
                                                   layer.input_scale,
                                                   batch_dim_padding=17)

            # Fused GEMM_DQ -- note we padded the input above because
            # torch._scaled_mm is more performant for matrices with
            # fake_qweight * batch dimension > 16.
            # Note that this could change in the future.
            output, _ = torch._scaled_mm(
                qinput,
                layer.weight,
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                bias=bias,
            )

        return torch.narrow(output, 0, 0, x.shape[0])


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def per_tensor_dequantize(
        tensor: torch.Tensor, inv_scale: Union[float,
                                               torch.Tensor]) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight
