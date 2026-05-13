#!/bin/bash

# ensure the script dies if any command fails, and that it fails if any command in a pipeline fails
set -eo pipefail

# source the build-env.sh script to get the environment variables
SCRIPT_DIR=$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)
source "$SCRIPT_DIR/build-env.sh"

# Install system packages
apt-get update && apt-get upgrade -y
apt-get install tesseract-ocr -y

# Install python dependencies
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install openai
pip install gradio

# Install vllm-gaudi
if [ -d "vllm-gaudi" ]; then
    echo "vllm-gaudi directory already exists. Removing it to ensure a clean installation."
    rm -rf vllm-gaudi
fi
git clone https://github.com/vllm-project/vllm-gaudi
cd vllm-gaudi
# checkout the specific commit for vllm-gaudi, or default to main if not set
export VLLM_GAUDI_COMMIT_HASH=${VLLM_GAUDI_COMMIT_HASH:-main}
echo "Using vLLM-Gaudi commit hash: $VLLM_GAUDI_COMMIT_HASH"
git checkout -- "$VLLM_GAUDI_COMMIT_HASH"

# set the VLLM_COMMIT_HASH to the default, if not already defined
export VLLM_COMMIT_HASH=${VLLM_COMMIT_HASH:-$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)}
echo "Using VLLM commit hash: $VLLM_COMMIT_HASH"
cd ..

# Build vLLM from source for empty platform, reusing existing torch installation
if [ -d "vllm" ]; then
    echo "vllm directory already exists. Removing it to ensure a clean installation."
    rm -rf vllm
fi
git clone https://github.com/vllm-project/vllm
cd vllm
git checkout -- "$VLLM_COMMIT_HASH"
pip install -r <(sed '/^torch/d' requirements/build.txt)
# note that newer versions have a different path to the requirements file
#pip install -r <(sed '/^torch/d' requirements/build/cuda.txt)
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
cd ..

cd vllm-gaudi
pip install -e .
cd ..

echo "VLLM and vLLM-Gaudi installed successfully. Starting the API server..."

# Start serving
cd vllm
VLLM_SKIP_WARMUP=true python -m vllm.entrypoints.openai.api_server \
--model $MODEL \
--max_model_len 99200 \
--tensor-parallel-size 8 \
--host $VLLM_HOST \
--port $VLLM_PORT \
--limit-mm-per-prompt '{"image":6}'  \
--enable-auto-tool-choice \
--tool-call-parser hermes \
--chat-template /app/chat_template.jinja
