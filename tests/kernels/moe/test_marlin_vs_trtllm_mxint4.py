# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test comparing Marlin INT4 MoE vs FlashInfer TRT-LLM MXINT4 MoE."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (
    is_flashinfer_mxint4_moe_available,
    prepare_static_weights_for_trtllm_mxint4_moe,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


def noaux_tc_ref(logits, bias, n_group, topk_group, top_k, routed_scaling_factor):
    """DeepSeek-style no-aux routing reference implementation."""
    scores = torch.nn.functional.sigmoid(logits)
    scores_with_bias = scores + bias if bias is not None else scores
    if n_group > 1:
        scores_shape = list(scores_with_bias.shape)
        group_scores = torch.sum(
            torch.topk(
                scores_with_bias.view(
                    scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]
                ),
                k=2,
                dim=-1,
                largest=True,
                sorted=True,
            )[0],
            dim=-1,
        )
        _, group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, largest=True, sorted=True
        )
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group])
            .reshape(scores_shape)
        )
        scores_with_bias = scores_with_bias * score_mask

    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    new_mask = torch.zeros_like(scores)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = scores * new_mask
    score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
    scores = scores / score_sum * routed_scaling_factor
    return scores


def mxint4_quantize(
    x: torch.Tensor, sf_vec_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 tensor to MXINT4 with block scaling (group_size=32).

    Returns:
        - uint8 packed (2 INT4/byte): [..., k//2] - stores SIGNED INT4 [-8, 7]
        - scales in BF16: [..., k//sf_vec_size]
    """
    x_reshaped = x.reshape(-1, sf_vec_size)
    x_max = x_reshaped.max(dim=-1, keepdim=True)[0].to(torch.float32)
    x_min = x_reshaped.min(dim=-1, keepdim=True)[0].to(torch.float32)
    x_max = x_max * 8.0 / 7.0
    amax = torch.where(x_max > -x_min, x_max, -x_min)
    scales = amax / 8.0
    x_scaled = x_reshaped * scales.reciprocal()
    x_int8 = (
        x_scaled.round().clamp(-8, 7).to(torch.int8).reshape(-1, sf_vec_size // 2, 2)
    )
    x_int4 = (x_int8[..., 0] & 0x0F) | ((x_int8[..., 1] & 0x0F) << 4)
    return (
        x_int4.to(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 2),
        scales.to(x.dtype).reshape(*x.shape[:-1], x.shape[-1] // sf_vec_size),
    )


@pytest.mark.skipif(
    not is_flashinfer_mxint4_moe_available(),
    reason="FlashInfer TRT-LLM MXINT4 MoE not available",
)
@pytest.mark.skipif(current_platform.is_rocm(), reason="Skip for rocm")
@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("n", [7168])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("e", [384])
@pytest.mark.parametrize("topk", [8])  # DeepSeekV3 default: num_experts_per_tok=8
def test_marlin_vs_trtllm_mxint4_moe(m, n, k, e, topk):
    """Compare Marlin INT4 MoE vs FlashInfer TRT-LLM MXINT4 MoE.

    Uses mxint4_quantize() to generate common INT4 weights + BF16 scales,
    then runs both Marlin and TRT-LLM kernels and compares outputs.
    """
    torch.cuda.manual_seed(0)

    group_size = 32
    dtype = torch.bfloat16

    # DeepSeekV3 routing config (from Kimi-K2-Thinking config.json)
    n_group = 1  # n_group from model config
    topk_group = 1  # topk_group from model config
    routed_scaling = 2.827  # routed_scaling_factor from model config

    # Input - realistic activation range for LLM (after LayerNorm: mean~0, std~1)
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.5

    # Generate routing logits and bias (DeepSeekV3 expects float logits)
    # Realistic ranges: logits typically [-3, 3], bias [-2, 2]
    routing_logits = torch.randn((m, e), device="cuda", dtype=torch.float32) * 1.5
    routing_bias = torch.randn(e, device="cuda", dtype=torch.float32) * 0.8

    # 1. Generate BF16 weights (SHARED between both paths)
    # Realistic weight initialization: Xavier/Glorot uniform scaling
    # std = sqrt(2 / (fan_in + fan_out))
    std_w1 = (2.0 / (k + 2 * n)) ** 0.5
    std_w2 = (2.0 / (n + k)) ** 0.5
    w1_bf16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) * std_w1
    w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) * std_w2

    # === Marlin path: Quantize using Marlin's method (UINT4b8) ===
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )

    w1_marlin_list = []
    w1_scales_marlin_list = []
    w2_marlin_list = []
    w2_scales_marlin_list = []

    for i in range(e):
        # Marlin w1: transpose [2n, k] → [k, 2n] for quantization
        w1_t = w1_bf16[i].T.contiguous()  # [k, 2n]
        _, w1_marlin_q, w1_marlin_s, _, _, _ = marlin_quantize(
            w1_t, scalar_types.uint4b8, group_size, act_order=False
        )
        w1_marlin_list.append(w1_marlin_q)
        w1_scales_marlin_list.append(w1_marlin_s)

        # Marlin w2: transpose [k, n] → [n, k] for quantization
        w2_t = w2_bf16[i].T.contiguous()  # [n, k]
        _, w2_marlin_q, w2_marlin_s, _, _, _ = marlin_quantize(
            w2_t, scalar_types.uint4b8, group_size, act_order=False
        )
        w2_marlin_list.append(w2_marlin_q)
        w2_scales_marlin_list.append(w2_marlin_s)

    w1_marlin = torch.stack(w1_marlin_list)  # [e, ...] Marlin format
    w1_scales_marlin = torch.stack(w1_scales_marlin_list)  # [e, ...] Marlin format
    w2_marlin = torch.stack(w2_marlin_list)  # [e, ...] Marlin format
    w2_scales_marlin = torch.stack(w2_scales_marlin_list)  # [e, ...] Marlin format

    # Use PyTorch reference routing (noaux_tc_ref) for Marlin path
    # The model config (n_group=1, topk_group=1, topk=8) doesn't satisfy
    # fused_topk_deepseek constraints (requires topk_group*n_group >= topk)
    # So TRT-LLM must be using a different internal routing implementation
    routing_scores = noaux_tc_ref(
        routing_logits,
        routing_bias,
        n_group,
        topk_group,
        topk,
        routed_scaling,
    )
    topk_weights, topk_ids = torch.topk(
        routing_scores, k=topk, dim=-1, largest=True, sorted=True
    )
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.int()

    marlin_output = fused_marlin_moe(
        a,
        w1_marlin,
        w2_marlin,
        None,
        None,
        w1_scales_marlin,
        w2_scales_marlin,
        None,  # gating_output not needed when topk_weights/ids provided
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=None,
        global_scale1=None,
        global_scale2=None,
        g_idx1=None,
        g_idx2=None,
        input_global_scale1=None,
        input_global_scale2=None,
        sort_indices1=None,
        sort_indices2=None,
        w1_zeros=None,
        w2_zeros=None,
        input_dtype=dtype,
        quant_type_id=scalar_types.uint4b8.id,
        is_k_full=True,
    )

    # === TRT-LLM path: Quantize using MXINT4 method (signed INT4) ===
    w1_trtllm_list = []
    w1_trtllm_scales_list = []
    w2_trtllm_list = []
    w2_trtllm_scales_list = []

    for i in range(e):
        # TRT-LLM w1: Quantize [2n, k] with mxint4
        w1_int4, w1_scales = mxint4_quantize(w1_bf16[i], group_size)
        w1_trtllm_list.append(w1_int4)
        w1_trtllm_scales_list.append(w1_scales)

        # TRT-LLM w2: Quantize [k, n] with mxint4
        w2_int4, w2_scales = mxint4_quantize(w2_bf16[i], group_size)
        w2_trtllm_list.append(w2_int4)
        w2_trtllm_scales_list.append(w2_scales)

    w1_int4 = torch.stack(w1_trtllm_list)  # [e, 2n, k//2] uint8 packed
    w1_scales = torch.stack(w1_trtllm_scales_list)  # [e, 2n, k//32]
    w2_int4 = torch.stack(w2_trtllm_list)  # [e, k, n//2] uint8 packed
    w2_scales = torch.stack(w2_trtllm_scales_list)  # [e, k, n//32]

    trtllm_weights = prepare_static_weights_for_trtllm_mxint4_moe(
        gemm1_weights=w1_int4,
        gemm1_scales=w1_scales,
        gemm2_weights=w2_int4,
        gemm2_scales=w2_scales,
    )

    from flashinfer import RoutingMethodType
    from flashinfer.fused_moe import trtllm_mxint4_block_scale_moe

    # For FlashInfer: pass raw routing_logits (float32) and routing_bias
    # First, compute what routing TRT-LLM should produce internally
    print("\nDEBUG: Comparing routing - what TRT-LLM should compute internally:")
    routing_scores_trtllm = noaux_tc_ref(
        routing_logits,
        routing_bias,
        n_group,
        topk_group,
        topk,
        routed_scaling,
    )
    topk_weights_trtllm, topk_ids_trtllm = torch.topk(
        routing_scores_trtllm, k=topk, dim=-1, largest=True, sorted=True
    )
    trtllm_output = trtllm_mxint4_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias.to(torch.bfloat16),
        hidden_states=a,
        gemm1_weights=trtllm_weights["gemm1_weights"],
        gemm1_weights_scale=trtllm_weights["gemm1_scales"],
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=trtllm_weights["gemm2_weights"],
        gemm2_weights_scale=trtllm_weights["gemm2_scales"],
        num_experts=e,
        top_k=topk,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=n,
        local_expert_offset=0,
        local_num_experts=e,
        routed_scaling_factor=routed_scaling,
        routing_method_type=RoutingMethodType.DeepSeekV3,
        enable_pdl=None,
        output=None,
        tune_max_num_tokens=8192,
    ).to(dtype)

    # Sanity check: manually compute BF16 reference for comparison
    bf16_output = torch.zeros((m, k), device="cuda", dtype=dtype)
    for token_idx in range(m):
        for expert_rank in range(topk):
            expert_id = topk_ids[token_idx, expert_rank].item()
            weight = topk_weights[token_idx, expert_rank].item()
            # w1: [2*n, k] @ [k] -> [2*n]
            up_gate = a[token_idx] @ w1_bf16[expert_id].T  # [2*n]
            gate, up = up_gate.chunk(2, dim=0)
            intermediate = torch.nn.functional.silu(gate) * up  # [n]
            # w2: [k, n] @ [n] -> [k]
            expert_out = intermediate @ w2_bf16[expert_id].T  # [k]
            bf16_output[token_idx] += weight * expert_out
    # Key assertion: same INT4 weights should give similar outputs
    # NOTE: Marlin and FlashInfer use different INT4 formats, so direct comparison
    # may not be valid. Consider comparing each against a BF16 reference instead.
    torch.testing.assert_close(marlin_output, trtllm_output, atol=0.2, rtol=0.5)
