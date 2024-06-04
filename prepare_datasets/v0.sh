#!/bin/bash

# 以下のデータセットを用意
# STAIR Captions(非商用利用画像除去)
# Japanese Visual Genome VQA

mkdir ./dataset/v0

# for Stage1
## STAIR Captions
### キャプション準備
git clone https://github.com/STAIR-Lab-CIT/STAIR-captions.git ./dataset/v0/STAIR-captions

mkdir ./dataset/v0/stair_captions_v1.2
tar -xzf ./dataset/v0/STAIR-captions/stair_captions_v1.2.tar.gz -C ./dataset/v0/stair_captions_v1.2
rm -rf ./dataset/v0/STAIR-captions

mkdir ./dataset/v0/LLaVA-Stair-Caption
python tools/stair_caption_to_llava_format.py

### 画像準備
wget -P ./dataset/v0 http://images.cocodataset.org/zips/train2014.zip
mkdir -p ./dataset/v0/images/stage1
unzip ./dataset/v0/train2014.zip -d ./dataset/v0/images/stage1
mv ./dataset/v0/images/stage1/train2014 ./dataset/v0/images/stage1/MS-COCO-train2014
rm ./dataset/v0/train2014.zip

cp ./dataset/v0/LLaVA-Stair-Caption/llava_stair_caption.json ./dataset/v0/llava_pretrain_stair.json

# Stage2
## Japanese Visual Genome VQA dataset
### キャプション準備
git clone https://github.com/yahoojapan/ja-vg-vqa.git ./dataset/v0/ja-vg-vqa
mkdir ./dataset/v0/VisualGenome
unzip ./dataset/v0/ja-vg-vqa/question_answers.json.zip -d ./dataset/v0/VisualGenome
rm -rf ./dataset/v0/ja-vg-vqa

wget -P ./dataset/v0/VisualGenome https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
unzip ./dataset/v0/VisualGenome/image_data.json.zip -d ./dataset/v0/VisualGenome
rm ./dataset/v0/VisualGenome/image_data.json.zip

python tools/visual_genome_to_llava_format.py

### 画像準備
mkdir -p ./dataset/v0/images/stage2
wget -P ./dataset/v0/images/stage2 https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -P ./dataset/v0/images/stage2 https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip ./dataset/v0/images/stage2/images.zip -d ./dataset/v0/images/stage2
unzip ./dataset/v0/images/stage2/images2.zip -d ./dataset/v0/images/stage2

rm ./dataset/v0/images/stage2/images.zip
rm ./dataset/v0/images/stage2/images2.zip