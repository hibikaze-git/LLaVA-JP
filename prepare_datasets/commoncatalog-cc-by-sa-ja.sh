#!/bin/bash

# 以下のデータセットを用意
# commoncatalog-cc-by-sa-ja

mkdir ./dataset/commoncatalog-cc-by-sa-ja

### キャプション準備
python tools/commoncatalog-cc-by-sa-ja/to_llava_format.py
