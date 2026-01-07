# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for MxInt4 + FlashInfer fused-MoE path"""

import torch

__all__ = [
    "prepare_static_weights_for_trtllm_mxint4_moe",
    "flashinfer_trtllm_mxint4_moe",
]

def prepare_static_weights_for_trtllm_mxint4_moe(
     gemm1_weights,
     gemm1_scales,
     gemm2_weights,
     gemm2_scales,
     num_experts,
):
    """
    Prepare MxInt4 weights for TRT-LLM kernel.
    
    Input:
        gemm1_weights: [num_experts, 2*intermediate_size, hidden_size//2] uint8
        gemm1_scales: [num_experts, 2*intermediate_size, hidden_size//32] bf16
        gemm2_weights: [num_experts, hidden_size, intermediate_size] uint8  
        gemm2_scales: [num_experts, hidden_size, intermediate_size//32] bf16
    
    Returns:
        Tuple of shuffled/packed weights and scales ready for kernel
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.fused_moe import (
        convert_to_block_layout, 
        block_scale_interleave,

    )
    from flashinfer.fp4_quantization import block_scale_interleave
    
    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}

    # Convert quantized weights to proper formats -
    gemm1_weights_mxint4 = gemm1_weights.view(torch.uint8)
    assert gemm1_scales.dtype == torch.bfloat16
    gemm2_weights_mxint4 = gemm2_weights.view(torch.uint8)
    assert gemm2_scales.dtype == torch.bfloat16

    epilogue_tile_m = 128
    gemm1_weights_mxint4_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_weights_mxint4_shuffled = []
    gemm2_scales_shuffled = []

    for i in range(num_experts):
            # Calculate the permute indices for the following:
            # 1. Reorder rows of W1 and scales for fused gated activation
            # 2. Shuffle weights and scaling factors for transposed mma output
            # for both w3_w1 and w2 weights and scale factors
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                _cache_permute_indices,
                gemm1_weights_mxint4[i],
                epilogue_tile_m,
            )
            gemm1_weights_shuffled = (
                gemm1_weights_mxint4[i]
                [permute_indices.to(gemm1_weights.device)]
                .contiguous()
            )
            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
                _cache_permute_indices,
                gemm1_scales[i],
                epilogue_tile_m,
                num_elts_per_sf=32,
            )
            gemm1_scales_shuffled.append(
                block_scale_interleave(
                    gemm1_scales[i]
                    [
                        permute_sf_indices.to(gemm1_scales.device)
                    ]
                    .contiguous()
                )
            )

            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                gemm2_weights_mxint4[i],
                epilogue_tile_m,
            )
            gemm2_weights_shuffled = (
                gemm2_weights_mxint4[i]
                [permute_indices.to(gemm2_weights.device)]
                .contiguous()
            )

            permute_sf_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                gemm2_scales[i],
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_shuffled.append(
                block_scale_interleave(
                    gemm2_scales[i]
                    [
                        permute_sf_indices.to(gemm2_scales.device)
                    ]
                    .contiguous()
                )
            )

            block_k = 128
            gemm1_weights_shuffled = convert_to_block_layout(
                gemm1_weights_shuffled, block_k
            )
            gemm2_weights_shuffled = convert_to_block_layout(
                gemm2_weights_shuffled.view(torch.uint8), block_k
            )

            gemm1_weights_mxint4_shuffled.append(gemm1_weights_shuffled)
            gemm2_weights_mxint4_shuffled.append(gemm2_weights_shuffled)

        gemm1_weights_mxint4_shuffled = torch.stack(gemm1_weights_mxint4_shuffled)
        gemm2_weights_mxint4_shuffled = torch.stack(gemm2_weights_mxint4_shuffled)
        gemm1_scales_shuffled = torch.stack(gemm1_scales_shuffled).view(torch.bfloat16)
        gemm2_scales_shuffled = torch.stack(gemm2_scales_shuffled).view(torch.bfloat16)

        return {
            "gemm1_weights": gemm1_weights_mxint4_shuffled,
            "gemm1_scales": gemm1_scales_shuffled,
            "gemm2_weights": gemm2_weights_mxint4_shuffled,
            "gemm2_scales": gemm2_scales_shuffled,
        } 


def flashinfer_trtllm_mxint4_moe(
    layer: torch.nn.Module,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    num_expert_group: int | None,
    topk_group: int | None,
    custom_routing_function: object | None,
    e_score_correction_bias: torch.Tensor | None,
) -> torch.Tensor:
    """
    Apply FlashInfer TensorRT-LLM MxInt4 MoE kernel.
    
    Simpler than FP4 version:
    - No input quantization needed (accepts bf16 directly)
    - No output scales
    - Scales are bf16 (not fp8)
    
    Args:
        layer: MoE layer with mxint4 weights
        x: Input tensor (bf16, no quantization needed!)
        router_logits: Router logits for expert selection
        top_k: Number of experts per token
        global_num_experts: Total number of experts
        num_expert_group: For grouped routing (DeepSeekV3)
        topk_group: Top-k within groups
        custom_routing_function: Custom routing (Llama4, etc.)
        e_score_correction_bias: Routing bias (DeepSeekV3)
    
    Returns:
        Output tensor from MoE layer
    """
    import flashinfer
    from vllm.model_executor.models.llama4 import Llama4MoE
    
    # NO input quantization needed - use bf16 directly
    assert x.dtype == torch.bfloat16
    
    # Determine routing method
    use_llama4_routing = custom_routing_function is Llama4MoE.custom_routing_function
    routing_method_type = layer.routing_method_type
    if use_llama4_routing:
        routing_method_type = flashinfer.RoutingMethodType.Llama4
    
    # Prepare routing bias
    routing_bias = None
    if e_score_correction_bias is not None:
        routing_bias = e_score_correction_bias.to(torch.bfloat16)
    
    # DeepSeekV3 requires float32 routing logits
    if routing_method_type == flashinfer.RoutingMethodType.DeepSeekV3:
        router_logits = router_logits.to(torch.float32)
    
    # Call MxInt4 kernel (simpler than FP4!)
    out = flashinfer.fused_moe.trtllm_mxint4_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=routing_bias,
        hidden_states=x,  # Direct bf16, no quantization!
        gemm1_weights=layer.w13_weight.data,
        gemm1_weights_scale=layer.w13_weight_scale.data,
        gemm1_alpha=layer.gemm1_alpha.data if hasattr(layer, 'gemm1_alpha') else None,
        gemm1_beta=layer.gemm1_beta.data if hasattr(layer, 'gemm1_beta') else None,
        gemm1_clamp_limit=layer.gemm1_clamp_limit.data if hasattr(layer, 'gemm1_clamp_limit') else None,
        gemm2_weights=layer.w2_weight.data,
        gemm2_weights_scale=layer.w2_weight_scale.data,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=num_expert_group if num_expert_group is not None else 0,
        topk_group=topk_group if topk_group is not None else 0,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type,
        enable_pdl=None,  # Auto-detect
        output=None,
        tune_max_num_tokens=8192,
    )[0]
    
    return out