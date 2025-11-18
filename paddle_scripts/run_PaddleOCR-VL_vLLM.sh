# Cuda12.8 + python 3.12 + PaddleOCR 3.3.2

# 0. System env config
export OMP_NUM_THREADS=1
unset LD_LIBRARY_PATH

# 0. Switch to venv
source /root/autodl-tmp/paddle-ocr-vl/bin/activate

# 0. Update pip
# python -m pip install --upgrade pip

# 1. Install PaddleOCR 3.3.2
# python -m pip install paddleocr[doc-parser]==3.3.2

# 3. Install spical Safetensors to support Paddle platform model
# python -m pip uninstall -y safetensors
# python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# 4. Install pre-build whl flash-attention
# [ -f /etc/network_turbo ] && source /etc/network_turbo    # autodl network speed up
# python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl
# unset http_proxy && unset https_proxy    # close network speed up

# 5. Install speed-up Platform
# paddleocr install_genai_server_deps vllm

# 6. Update flash-attention to 2.8.3
# python -m pip install flash-attn==2.8.3

# 7. Run vllm server
paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8118 --backend_config /root/paddle_scripts/vllm_config.yaml