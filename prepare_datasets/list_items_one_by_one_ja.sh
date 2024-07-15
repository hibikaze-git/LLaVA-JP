#!/bin/bash

# 以下のデータセットを用意
# list_items_one_by_one_ja

mkdir ./dataset/list_items_one_by_one_ja

### キャプション準備
python tools/list_items_one_by_one_ja/to_llava_format_detailed.py
