#!/bin/bash

# 以下のデータセットを用意
# wit_base

mkdir ./dataset/wit_base

### キャプション準備
python tools/wit_base/to_llava_format.py