#!/bin/bash

cwd=$PWD
MUJOCO_SOURCE=$cwd/mujoco-source
MUJOCO_BUILD=$MUJOCO_SOURCE/mujoco-build
MUJOCO_INSTALL=$cwd/mujoco-install
mkdir $MUJOCO_INSTALL

git clone https://github.com/google-deepmind/mujoco.git $MUJOCO_SOURCE
mkdir $MUJOCO_BUILD && cd $MUJOCO_BUILD
cmake .. -DCMAKE_INSTALL_PREFIX=$MUJOCO_INSTALL
make -j5
make install

cd ../python
python3 -m venv /tmp/env
source /tmp/env/bin/activate
bash make_sdist.sh
deactivate
MUJOCO_PATH=$MUJOCO_INSTALL MUJOCO_PLUGIN_PATH=$MUJOCO_SOURCE/plugin pip install dist/mujoco*.tar.gz
