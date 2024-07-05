#!/bin/bash

# 以下のデータセットを用意
# commoncatalog-cc-by-ext

mkdir ./dataset/commoncatalog-cc-by-ext

### キャプション準備
python tools/commoncatalog-cc-by-ext/to_llava_format.py
