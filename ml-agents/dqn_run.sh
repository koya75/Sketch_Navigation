#!/bin/bash
set -e
set -u

USERNAME=$(whoami)
docker run -it \
	--mount type=bind,source="/home/koya/hoge/MapNav",target=/home/${USERNAME}/ML-Agents/ \
	--mount type=bind,source="/home/koya/HSR_Unity",target=/home/${USERNAME}/ML-Agents/dqn/ml-agents \
	--mount type=bind,source="/data1/honda/results/DQN",target=/home/${USERNAME}/ML-Agents/dqn/DQN \
	--user=$(id -u $USER):$(id -g $USER) \
	--env="DISPLAY" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--rm \
	--gpus=all \
	--workdir=/home/${USERNAME}/ML-Agents/dqn \
	--name=${USERNAME}_ml_agents_dqn_2 ${USERNAME}/unity /bin/bash