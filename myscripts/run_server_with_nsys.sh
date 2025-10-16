#!/bin/bash
set +x

export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND="latency" 
export VLLM_ATTENTION_BACKEND="FLASHINFER_MLA"
export VLLM_PROFILE_ITERATION_INFO=true

TP_SIZE=8
MAX_MODEL_LEN=2048

DATE=$(date +%Y%m%d%H%M%S)
PROFILE_NAME="vllm_profile_${DATE}"

# Start server with nsys profiling
# nsys profile --output=${PROFILE_NAME} --delay=120 --trace=cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true \
vllm serve /models/nvidia-DeepSeek-R1-0528-FP4/ \
  --tensor-parallel-size ${TP_SIZE} --max-model-len ${MAX_MODEL_LEN} \
  --no-enable-prefix-caching --host 0.0.0.0 --port 8000 --kv-cache-dtype fp8 --max-num-seqs 128 \
  --hf-overrides '{"num_hidden_layers": 4}'


#server

python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8087 --model nvidia/DeepSeek-R1-0528-FP4-v2 --tokenizer nvidia/DeepSeek-R1-0528-FP4-v2 --dtype auto --kv-cache-dtype fp8 --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel --swap-space 16 --max-num-seqs 512 --trust-remote-code --max-model-len 10240 --gpu-memory-utilization 0.9 --max-num-batched-tokens 1600 --no-enable-prefix-caching --async-scheduling --compilation_config.pass_config.enable_fi_allreduce_fusion true --compilation_config.pass_config.enable_attn_fusion true --compilation_config.pass_config.enable_noop true --compilation_config.custom_ops+=+quant_fp8,+rms_norm --cuda_graph_sizes 2048 --compilation_config.cudagraph_mode FULL_DECODE_ONLY --compilation_config.splitting_ops []
	
# client
python3 /lustre/fsw/coreai_comparch_infbench/shuhaoy/1007-test/vllm_fp4/vllm-results/fp4/1600/profile/dep8_t/vLLMSweep_20251007-194250/coreai_comparch_trtllmvllm-server-client/num_gpus.1.num_nodes.1/benchmark_serving.py --backend vllm --host 0.0.0.0 --port 8087 --model nvidia/DeepSeek-R1-0528-FP4-v2 --num-prompts 10240 --trust-remote-code --ignore-eos --max-concurrency 2048 --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 --dataset-name random --save-result --result-filename sa_benchmark_serving_results.json
	
	
# env	
export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA	
export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'	
export VLLM_USE_FLASHINFER_MOE_FP4=1	
export VLLM_FLASHINFER_MOE_BACKEND=throughput	
export FORCE_NUM_KV_SPLITS=1	
	
# for nsys	
# export VLLM_PROFILE_START_STOP="2100-2120,2600-2620"	