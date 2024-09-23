# syntax = devthefuture/dockerfile-x
# SPDX-License-Identifier: MIT

RUN apt-get update && \
        apt-get install --no-install-recommends -y \
        git \
        python3-venv \
        libgl1-mesa-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxrandr-dev \
        libxi-dev \
        build-essential \
        cmake \
        ninja-build && \
        apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/google-deepmind/mujoco.git /tmp/mujoco && \
        mkdir /tmp/mujoco/build && \
        cd /tmp/mujoco/build && \
        cmake .. -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/mujoco -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=TRUE && \
        ninja && \
        ninja install

RUN cd /tmp/mujoco/python && \
        python3 -m venv /tmp/env && \
        . /tmp/env/bin/activate && \
        bash make_sdist.sh && \
        deactivate && \
        MUJOCO_PATH=/opt/mujoco MUJOCO_PLUGIN_PATH=../../plugin pip install dist/mujoco*.tar.gz && \
        rm -rf /tmp/mujoco
