#!/bin/sh
'''
# Using Docker image: seokg1023/vml-pytorch:vessl
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.2 LTS
Release:        20.04
Codename:       focal

#! this shell file should be copied from data2/YerimShin/deeprl_ws into /workspace/yerim/deeprl_ws/set_envs.sh
'''

# # install sudo 
# apt-get update && \
#       apt-get -y install sudo

# # 가능 드라이버 확인
# sudo apt search nvidia-driver
# 

# before running shell, 
# $ chmod +x $filename.sh 
# $ cd /root 


TARGET_WORKING_DIR="/data/yerim/deeprl_ws"
SOURCE_ISAAC_DIR="/data2/YerimShin/deeprl_ws"
CONDA_ENVS="/opt/conda/envs/"
ASE_ENV="cml"


# copy isaac gym from intra to server
ISAAC_DIR="$SOURCE_ISAAC_DIR/IsaacGym_Preview_4_Package"
if [ -d "$ISAAC_DIR" ]
then
    echo  "$ISAAC_DIR is a directory."
fi

# conda activate
conda init bash
source ~/.bashrc
source /opt/conda/etc/profile.d/conda.sh

#cp -r "$SOURCE_ISAAC_DIR/IsaacGym_Preview_4_Package" $TARGET_WORKING_DIR
# create conda environment for isaacgym
cd $TARGET_WORKING_DIR/IsaacGym_Preview_4_Package/isaacgym
./create_conda_env_rlgpu.sh

# install isaac gym in new conda env
conda activate $ASE_ENV
cd $TARGET_WORKING_DIR/IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .

# set environment setting for ase -> later again
cd $CONDA_ENVS$ASE_ENV
conda activate $ASE_ENV
set +H
touch ./etc/conda/activate.d/env_vars.sh
cp $SOURCE_ISAAC_DIR/env_vars/act_env_vars.sh $CONDA_ENVS$ASE_ENV/etc/conda/activate.d/env_vars.sh

touch ./etc/conda/deactivate.d/env_vars.sh
cp $SOURCE_ISAAC_DIR/env_vars/deact_env_vars.sh $CONDA_ENVS$ASE_ENV/etc/conda/deactivate.d/env_vars.sh

# download IsaacGymEnv from git
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git $TARGET_WORKING_DIR/IsaacGymEnvs

# download ASE from git
git clone https://github.com/haaappytoast/ASE.git $TARGET_WORKING_DIR/ASE
# install requirements in ase conda env
# conda activate $ASE_ENV

# python -m pip install torch==1.8.1 \
# numpy==1.21.1 \
# termcolor==1.1.0 \
# rl-games==1.1.4 \
# tensorboard==1.15.0

cd $TARGET_WORKING_DIR
source start_code.sh
