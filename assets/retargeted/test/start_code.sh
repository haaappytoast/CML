#!/bin/sh

# go into working directory
TARGET_WORKING_DIR="/workspace/yerim/deeprl_ws"
ASE_ENV="ase"
conda activate $ASE_ENV

pip install torch==1.8.1
pip install numpy==1.21.1
pip install termcolor==1.1.0
pip install rl-games==1.1.4
pip install tensorboard==1.15.0

cd $TARGET_WORKING_DIR/ASE

git init
git remote update
git checkout remotes/origin/temp



