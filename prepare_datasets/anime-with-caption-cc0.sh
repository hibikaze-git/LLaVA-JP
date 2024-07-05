#!/bin/bash

# 以下のデータセットを用意
# anime-with-caption-cc0

mkdir -p ./dataset/anime-with-caption-cc0/images

### キャプション準備
python tools/anime-with-caption-cc0/to_llava_format.py
