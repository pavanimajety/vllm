import huggingface_hub
import nvtx

from vllm import LLM, SamplingParams


import os

os.environ["FORCE_NUM_KV_SPLITS"] = "1"
# os.environ['VLLM_USE_FLASHINFER_MOE_FP4'] = "1"
# os.environ['VLLM_FLASHINFER_MOE_BACKEND'] = "latency"

# os.environ['VLLM_ATTENTION_BACKEND'] = "CUTLASS_MLA"
# os.environ['VLLM_ATTENTION_BACKEND'] = "FLASHINFER"
# os.environ['VLLM_FLASH_ATTN_VERSION'] = "2"

# os.environ['VLLM_FLASHINFER_USE_RAGGED'] = "1"
# os.environ['VLLM_FLASHINFER_USE_RAGGED'] = "true"
# os.environ['VLLM_DISABLE_FLASHINFER_PREFILL'] = "1"
# os.environ['VLLM_MLA_DISABLE'] = "1"
# os.environ['VLLM_MULTIPROC_METHOD'] = "spawn"
# os.environ['VLLM_LOGGING_LEVEL'] = "DEBUG"

# model_str = "meta-llama/Llama-2-7b-hf"
# model_str = "meta-llama/Llama-3.1-8B"

# model_str = "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV"
# model_str =  "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
# model_str = "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ" 

# with nvtx.annotate("initialization", color="blue"):
# model_str = '/workspace/scratch-pmaj/vllm/ckpts/MO/llama3p1-8B-instruct-int4-awq'
# model_str = "/workspace/scratch-pmaj/vllm/ckpts/llama3p1-8b-instruct-nvfp4"
# model_str=  "nvidia/Llama-3.1-8B-Instruct-FP8"
# model_str = "deepseek-ai/DeepSeek-V2-Lite"
# model_str = "/models/DSR1-FP8"
# model_str = "/models/nvidia-DeepSeek-R1-0528-FP4"
# model_str=  "nvidia/Llama-3.3-70B-Instruct-FP4"
# model_str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# model_str = "/models/nvidia-DeepSeek-R1-0528-FP4/"
# model_str = "nvidia/DeepSeek-R1-0528-FP4-v2"
model_str = "nvidia/DeepSeek-R1-0528-FP4" # has kv scales, need to be loaded properly and the correct kv-cache-dtype must be set.

# model_str = "openai/gpt-oss-120b"
# vllm serve serve amd/Llama-3.1-8B-Instruct-FP8-KV -O '{"pass_config":{"enable_attn_fusion": true}}'
# with nvtx.annotate("context_generate", color="green"):
# model_str = "meta-llama/Llama-3.1-8B-Instruct"
model = LLM(model=model_str,
                tensor_parallel_size=8,
                gpu_memory_utilization=0.8,
                max_model_len=2048,
                kv_cache_dtype="fp8",
                # compilation_config={},
                # data_parallel_size=4,
                # enable_expert_parallel=True,
                # kv_cache_dtype="fp8",
                # max_num_batched_tokens=8,
                # quantization="modelopt_fp4",
                # enforce_eager=True,
                # compilation_config={"pass_config":{"enable_attn_fusion":True}},
                compilation_config={
                                    # "custom_ops":["+rotary_embedding"],
                                    "debug_dump_path":"./.debug_dump_no_rotary",
                                    # "cache_dir":"./.torch_compile_cache",
                                    },
                # compilation_config={"custom_ops":["+silu_and_mul"],"pass_config":{"enable_fusion":True,"enable_noop":True}},
                # quantization="nvfp4",
                # quantization="fp8",
                # # kv_cache_dtype="fp8",
                # enforce_eager=True,
                # max_model_len=8192
                )
# model = LLM(model=model_str)
params = SamplingParams(temperature=0)
#
prompts = [
    "Hello, my name is", "The president of the United States is",
    "The capital of France is", "New york times is great at one",
    "The future holds infinite possibilities for many of us who do not think about what ever comes next except the moment we are in. "
]
# with nvtx.annotate("context_generate", color="green"):
result = model.generate(prompts=prompts, sampling_params=params)
for output in result:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(
        f"\n\n Prompt: {prompt!r}, \nGenerated text: {generated_text!r}, \ntoken_ids: {output.outputs[0].token_ids}"
    )
