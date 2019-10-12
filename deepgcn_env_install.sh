#!/usr/bin/env bash
# make sure command is : source deepgcn_env_install.sh

# install anaconda3.
# cd ~/
# wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
# bash Anaconda3-2019.07-Linux-x86_64.sh

# module load, uncommet if using local machine
#module purge
#module load gcc
#module load cuda/10.1.105
#
# make sure your annaconda3 is added to bashrc
#source activate
#source ~/.bashrc

conda create -n deepgcn
conda activate deepgcn
conda install -y pytorch torchvision cudatoolkit=10.0 tensorflow python=3.7 -c pytorch

# uncomment if you use your own cuda (not by module load) or uncomment cuda path is already in ~/.bashrc
# export PATH=/usr/local/cuda/bin:$PATH
# export CPATH=/usr/local/cuda/include:$CPATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

# command to install pytorch geometric
pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv
pip install torch-geometric

# install useful modules
pip install requests # sometimes pytorch geometric forget to install it, but it is in need
pip install tqdm

