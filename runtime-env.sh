#!/bin/bash

# Change the model name to the one you want to use. You can find the model name on Hugging Face Model Hub at:
# https://huggingface.co/models
export MODEL="Qwen/Qwen3-VL-235B-A22B-Thinking"
#export MODEL="Qwen/Qwen3-VL-235B-A22B-Instruct"
#export MODEL="Qwen/Qwen2.5-VL-32B-Instruct"

export VLLM_HOST=0.0.0.0
export VLLM_PORT=8001
