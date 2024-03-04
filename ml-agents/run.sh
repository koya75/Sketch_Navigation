#!/bin/bash
set -e
set -u

USERNAME=$(whoami)
docker run -it \
	--mount type=bind,source="$(pwd)",target=/home/${USERNAME}/ML-Agents/ \
	--user=$(id -u $USER):$(id -g $USER) \
	--env="DISPLAY" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--rm \
	--gpus=all \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	--workdir=/home/${USERNAME}/ML-Agents \
	--name=${USERNAME}_ml_agents ${USERNAME}/unity /bin/bash