# Cuda 12.8 + Python 3.12 + Paddle 3.1.1
# Paddle 3.2.1 npt support Paddlex High-Performance Inference

# 0. System env config
export OMP_NUM_THREADS=1
unset LD_LIBRARY_PATH

# 0. Switch to venv
source /root/autodl-tmp/paddlex-hpi/bin/activate

# 0. Update pip
# python -m pip install --upgrade pip

# 1. Install paddle-gpu & paddlex
# python -m pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# python -m pip install paddlex[base,ocr,multimodal,serving,genai-client]==3.3.9

# 2. Install paddlex serving & hpi-gpu
# python -m pip uninstall -y numpy matplotlib
# python -m pip install numpy==1.26.4 matplotlib==3.8.4

# 3. Install Paddlex Plugin
# paddlex --install serving
# paddlex --install hpi-gpu
# paddlex --install paddle2onnx

# 4. Install spical Safetensors to support Paddle platform model
# python -m pip uninstall -y safetensors
# python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# 5. Create PaddleOCR-VL config
# paddlex --get_pipeline_config PaddleOCR-VL

# 6. Run PaddleOCR-VL pipeline
paddlex --serve --pipeline /root/paddle_scripts/PaddleOCR-VL.yaml --host '127.0.0.1' --port 6006 --device gpu:0 --use_hpi --hpi_config