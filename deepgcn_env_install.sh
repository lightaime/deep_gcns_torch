#!/usr/bin/env bash
# install anaconda3.
cd ~/
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh
source ~/.bashrc

# updata conda
conda update conda
conda create -n deepgcn
conda activate deepgcn
conda install -y pytorch torchvision cudatoolkit=10.0 tensorflow -c pytorch

# export PATH=/usr/local/cuda/bin:$PATH
# export CPATH=/usr/local/cuda/include:$CPATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv
pip install torch-geometric
pip install tqdm
