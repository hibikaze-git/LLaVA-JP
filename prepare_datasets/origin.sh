#!/bin/bash

# for Stage1
## STAIR Captions
### キャプション準備
git clone https://github.com/STAIR-Lab-CIT/STAIR-captions.git ./dataset/STAIR-captions

mkdir ./dataset/stair_captions_v1.2
tar -xzf ./dataset/STAIR-captions/stair_captions_v1.2.tar.gz -C ./dataset/stair_captions_v1.2
rm -rf ./dataset/STAIR-captions

mkdir ./dataset/LLaVA-Stair-Caption
python tools/stair_caption_to_llava_format.py

### 画像準備
wget -P ./dataset http://images.cocodataset.org/zips/train2014.zip
mkdir -p ./dataset/images/stage1
unzip ./dataset/train2014.zip -d ./dataset/images/stage1
mv ./dataset/images/stage1/train2014 ./dataset/images/stage1/MS-COCO-train2014
rm ./dataset/train2014.zip


## LLaVA-CC3M-Pretrain-595K-JA
### キャプション準備
mkdir ./dataset/LLaVA-CC3M-Pretrain-595K
python tools/create_chat_ja_calm2.py

### 画像準備
wget -P ./dataset https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip?download=true
mv ./dataset/images.zip?download=true ./dataset/images.zip
mkdir ./dataset/images/stage1/LLaVA-CC3M-Pretrain-595K
unzip ./dataset/images.zip -d ./dataset/images/stage1/LLaVA-CC3M-Pretrain-595K
rm ./dataset/images.zip

## Stage1のキャプションをマージ
python tools/merge_data.py


# Stage2
## Japanese Visual Genome VQA dataset
### キャプション準備
git clone https://github.com/yahoojapan/ja-vg-vqa.git ./dataset/ja-vg-vqa
mkdir ./dataset/VisualGenome
unzip ./dataset/ja-vg-vqa/question_answers.json.zip -d ./dataset/VisualGenome
rm -rf ./dataset/ja-vg-vqa

wget -P ./dataset/VisualGenome https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
unzip ./dataset/VisualGenome/image_data.json.zip -d ./dataset/VisualGenome
rm ./dataset/VisualGenome/image_data.json.zip

python tools/visual_genome_to_llava_format.py

### 画像準備
mkdir -p ./dataset/images/stage2
wget -P ./dataset/images/stage2 https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -P ./dataset/images/stage2 https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip ./dataset/images/stage2/images.zip -d ./dataset/images/stage2
unzip ./dataset/images/stage2/images2.zip -d ./dataset/images/stage2

rm ./dataset/images/stage2/images.zip
rm ./dataset/images/stage2/images2.zip