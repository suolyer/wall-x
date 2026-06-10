#!/bin/bash

export ENABLE_CUDA_GRAPH=True
export ENABLE_EXPERIMENTAL_INFERENCE_ENGINE=True

CKPT_PATH=/path/to/checkpoint

python -m wall_x._vendor.harrix.serving.launch_serving \
  --env X2ROBOT \
  --port 32195 \
  model-config:server-model-config \
  --model-config.checkpoint-path "$CKPT_PATH" \
  --model-config.action-horizon 32 \
  --model-config.robot-action-interpolate-multiplier 1 \
  --model-config.robot-action-end-ratio 1.0 \
  --model-config.robot-type desktop
