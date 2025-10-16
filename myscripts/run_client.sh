#!/bin/bash
set +x

INPUT_LEN=1024
OUTPUT_LEN=1024
NUM_PROMPTS=128

# Wait for server to be ready
echo "Waiting for server to be ready..."
while ! curl -s http://0.0.0.0:8000/health > /dev/null; do
  echo "Server not ready, waiting 5 seconds..."
  sleep 5
done
echo "Server is ready!"

vllm bench serve \
  --backend vllm \
  --model /models/nvidia-DeepSeek-R1-0528-FP4/ \
  --num-prompts ${NUM_PROMPTS} \
  --dataset-name random \
  --random-input ${INPUT_LEN} \
  --random-output ${OUTPUT_LEN}
