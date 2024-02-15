#!/bin/sh
# cd /data2/YerimShin/deeprl_ws/
# ./run_init.sh

# conda activate
conda init bash
source ~/.bashrc

#conda update -n base conda

TARGET_WORKING_DIR="/workspace/yerim/deeprl_ws"
SOURCE_ISAAC_DIR="/data2/YerimShin/deeprl_ws"
CONDA_ENVS="/opt/conda/envs/"
ASE_ENV="ase"

# copy isaac gym from intra to server
cp $SOURCE_ISAAC_DIR/set_envs.sh $TARGET_WORKING_DIR/set_envs.sh
cp $SOURCE_ISAAC_DIR/start_code.sh $TARGET_WORKING_DIR/start_code.sh

# create conda environment for isaacgym
cd $TARGET_WORKING_DIR
./set_envs.sh
