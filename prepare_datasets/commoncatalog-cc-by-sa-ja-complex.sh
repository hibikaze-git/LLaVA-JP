#!/bin/bash

# 以下のデータセットを用意
# commoncatalog-cc-by-sa-ja-complex

mkdir ./dataset/commoncatalog-cc-by-sa-ja-complex

### キャプション準備
python tools/commoncatalog-cc-by-sa-ja-complex/to_llava_format.py
