#!/bin/bash
# Download Qwen3-VL-4B-Instruct (ModelScope) and build a separate venv for it (the openpi venv pins transformers 4.53).
#   nohup bash setup_qwen.sh > /data_cf/qwen/setup.log 2>&1 &
set -u
Q=/data_cf/qwen
mkdir -p $Q/models
echo "[$(date +%H:%M:%S)] download start"
/usr/bin/python3 - <<'EOF'
from modelscope import snapshot_download
p = snapshot_download("Qwen/Qwen3-VL-4B-Instruct", cache_dir="/data_cf/qwen/models")
print("MODEL_DIR", p)
EOF
echo "[$(date +%H:%M:%S)] download rc=$?"
du -sh $Q/models
echo "[$(date +%H:%M:%S)] venv start"
uv venv --python 3.11 $Q/venv
uv pip install --python $Q/venv/bin/python \
  "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install --python $Q/venv/bin/python \
  "transformers==4.57.1" "accelerate==1.10.1" "peft==0.17.1" "qwen-vl-utils==0.0.14" \
  "pillow==11.3.0" "numpy==2.2.6" "safetensors==0.6.2" "tensorboard==2.20.0"
$Q/venv/bin/python -c "import torch,transformers,peft;print('VENV_OK',torch.__version__,transformers.__version__,peft.__version__,torch.cuda.is_available())"
echo "[$(date +%H:%M:%S)] all done"
