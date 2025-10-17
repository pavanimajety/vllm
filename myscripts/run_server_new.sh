#!/bin/bash
set +x

#export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA	
export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'	
export VLLM_USE_FLASHINFER_MOE_FP4=1	
export VLLM_FLASHINFER_MOE_BACKEND=throughput	
export FORCE_NUM_KV_SPLITS=1
#export VLLM_PROFILE_ITERATION_INFO=true 
export VLLM_PROFILE_START_STOP="2100-2120,2600-2620"
NSYS_OUT_NAME=1024-1024-512-EP8-FP4-CUTLASS-MLA-FP8KV-FI-CUTLASS-MOE
nsys profile \
    -t cuda,nvtx \
    -c cudaProfilerApi --capture-range-end="repeat[:2]" \
    --cuda-graph-trace=node \
    --trace-fork-before-exec=true \
    -o ${NSYS_OUT_NAME} -f true \
    vllm serve --host 0.0.0.0 --port 8087 --model nvidia/DeepSeek-R1-0528-FP4 --dtype  auto --kv-cache-dtype fp8 --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1 --enable-expert-parallel --swap-space 16 --max-num-seqs 512 --trust-remote-code --max-model-len 10240 --gpu-memory-utilization 0.9 --max-num-batched-tokens 1600 --no-enable-prefix-caching --async-scheduling --compilation_config.pass_config.enable_fi_allreduce_fusion true --compilation_config.pass_config.enable_attn_fusion true --compilation_config.pass_config.enable_noop true --compilation_config.custom_ops+=+quant_fp8,+rms_norm --cuda_graph_sizes 2048 --compilation_config.cudagraph_mode FULL_DECODE_ONLY --compilation_config.splitting_ops []


# --gpu-metrics-devices=0 : To only collect metrics for GPU 0
#    --gpu-metrics-frequency=50000 \
