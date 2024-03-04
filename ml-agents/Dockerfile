# From https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN yes | unminimize

RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-xenial main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN wget https://packages.cloud.google.com/apt/doc/apt-key.gpg && apt-key add apt-key.gpg
RUN apt-get update && \
  apt-get install -y --no-install-recommends wget curl tmux vim git gdebi-core \
  build-essential python3-pip unzip google-cloud-sdk htop mesa-utils xorg-dev xorg \
  libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev xvfb libglib2.0-0 libsm6 libxrender1 libxext6 \
  tzdata language-pack-ja-base language-pack-ja fonts-ipafont-gothic dbus-x11 fcitx-mozc fcitx-imlist wget curl ssh zsh terminator gnome-terminal git vim tig \ 
  dbus-x11 libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 nodejs-dev node-gyp libssl1.0-dev \
  gconf-service lib32gcc1 lib32stdc++6 libasound2 libc6 libc6-i386 libcairo2 libcap2 libcups2 libdbus-1-3 \
  libexpat1 libfontconfig1 libfreetype6 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libgl1-mesa-glx libgl1 libglib2.0-0 \
  libglu1-mesa libglu1 libgtk2.0-0 libnspr4 libnss3 libpango1.0-0 libstdc++6 libx11-6 libxcomposite1 libxcursor1 \
  libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxtst6 zlib1g debconf npm fuse nautilus && \
  wget http://security.ubuntu.com/ubuntu/pool/main/libx/libxfont/libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb && \
  yes | gdebi libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb


RUN python3 -m pip install --upgrade pip
RUN pip install setuptools==41.0.0

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

#checkout ml-agents for SHA
RUN mkdir /ml-agents
WORKDIR /ml-agents
ARG SHA
RUN git init
RUN git remote add origin https://github.com/Unity-Technologies/ml-agents.git
RUN git fetch --depth 1 origin $SHA
RUN git checkout FETCH_HEAD

ARG USERNAME=user
ARG UID=1001
RUN useradd  --create-home -m -s /bin/bash -u $UID $USERNAME
USER $USERNAME

ENV PATH="/home/$USERNAME/.local/bin:$PATH"

RUN pip install mlagents==0.26.0
RUN pip install mlagents_envs==0.26.0
RUN pip install scikit-image

# get untiy-hub
RUN mkdir -p /opt/unity \
  && wget -P /opt/unity https://public-cdn.cloud.unity3d.com/hub/prod/UnityHub.AppImage \
  && chown -R ${USER}:${GID} /opt/unity && chmod +x /opt/unity/UnityHub.AppImage