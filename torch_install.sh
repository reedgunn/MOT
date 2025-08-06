#!/bin/bash
set -e

echo "Activating Python environment..."
# source ~/MOT/venv/bin/activate  # Uncomment if needed

echo "Installing PyTorch 2.1.0 (NVIDIA build)..."
pip install https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

echo "Checking out torchvision v0.16.0 and installing from source..."
cd ~/Tools/vision
git checkout v0.16.0
python setup.py clean
python setup.py install

echo "Checking out torchaudio v2.1.0 and installing from source..."
cd ~/Tools/audio
git checkout v2.1.0
python setup.py clean
python setup.py install

echo "Verifying installs..."
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)"

echo "✅ Installation complete."
