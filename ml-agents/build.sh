#!/bin/bash
set -e
set -u

username=$(whoami)
uid=$(id -u)

docker build --network host -t ${username}/unity \
    --build-arg USERNAME=${username} \
    --build-arg UID=${uid} \
    -f Dockerfile \
    .
