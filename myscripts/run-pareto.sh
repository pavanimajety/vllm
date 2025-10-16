#!/bin/bash
set +x

export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND="latency" 
export VLLM_ATTENTION_BACKEND="FLASHINFER_MLA"

INPUT_LEN=1024
OUTPUT_LEN=1024
NUM_ITERS=50
TP_SIZE=8
MAX_MODEL_LEN=2048

DATE=$(date +%Y%m%d%H%M%S)
PROFILE_NAME="vllm_profile_${DATE}"
nsys profile --output=${PROFILE_NAME} --delay=100 --duration=30 --trace=cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true \
  vllm bench latency --model /models/nvidia-DeepSeek-R1-0528-FP4/ \
 --tensor-parallel-size ${TP_SIZE} --enable-expert-parallel --max-model-len ${MAX_MODEL_LEN} \
  --no-enable-prefix-caching  --num-iters ${NUM_ITERS} --input-len ${INPUT_LEN} --output-len ${OUTPUT_LEN} --kv-cache-dtype fp8