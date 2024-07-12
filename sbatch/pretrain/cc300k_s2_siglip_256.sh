#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-19
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/train_llava-jp_stage1.out
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --mem=200GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

bash scripts/pretrain/pretrain_accelerate.sh \
    ./configs/train/pretrain/base.json \
    ./configs/image_encoder/siglip-base-patch16-256-multilingual.json \
    ./configs/dataset/cc300k.json \
    ./configs/model/tanuki-8b.json \
    ./output_llava/checkpoints/pretrain-llava-jp-Tanuki-8B-vision-cc300k-s2_siglip_256 \
    llava-jp-stage1 \
    Tanuki-8B-vision-cc300k-s2_siglip_256
