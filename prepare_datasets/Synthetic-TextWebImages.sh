#!/bin/bash

# 以下のデータセットを用意
# Synthetic-TextWebImages

mkdir -p ./dataset/Synthetic-TextWebImages/ner-wikipedia-dataset/images
mkdir -p ./dataset/Synthetic-TextWebImages/wikipedia-22-12-ja-embeddings/images

### キャプション準備
python tools/Synthetic-TextWebImages/to_llava_format_ner.py
python tools/Synthetic-TextWebImages/to_llava_format_wiki.py
