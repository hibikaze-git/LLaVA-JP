#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-1
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/train_llava-jp_stage1.out
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

bash scripts/pretrain/pretrain_accelerate.sh \
    ./configs/train/pretrain/base_mlp6x_gelu.json \
    ./configs/image_encoder/siglip-so400m-patch14-384.json \
    ./configs/dataset/v1_stage_1.json \
    ./configs/model/tanuki-8b.json \
    ./output_llava/checkpoints/pretrain-llava-jp-Tanuki-8B-vision-v1_mlp6x_gelu \
    llava-jp-stage1 \
    Tanuki-8B-vision-v1_mlp6x_gelu
