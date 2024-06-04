#!/bin/sh

if [ ! -d "/tmp/own/docker/entrypoint/check" ]; then
    mkdir /tmp/own/docker/entrypoint/check
    echo 'export HF_HUB_CACHE=/home/docker/LLaVA-JP/cache' >> ~/.bashrc
    exec "$@"
else
    exec "$@"
fi