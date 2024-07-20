#!/bin/bash

# 以下のデータセットを用意
# TextOCR

mkdir -p ./dataset/TextOCR

wget -P ./dataset/TextOCR https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json
wget -P ./dataset/TextOCR https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

unzip ./dataset/TextOCR/train_val_images.zip -d ./dataset/TextOCR

rm -rf ./dataset/TextOCR/train_val_images.zip

### キャプション準備
python tools/TextOCR/to_llava_format.py
