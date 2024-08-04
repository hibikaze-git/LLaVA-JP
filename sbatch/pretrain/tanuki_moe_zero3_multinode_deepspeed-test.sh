#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-[6,17]
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/train_llava-jp_stage1.out
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1500GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp-deepspeed

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

bash scripts/pretrain/pretrain_deepspeed_zero3_multinode.sh \
    ./configs/train/pretrain/base_tanuki_moe.json \
    ./configs/image_encoder/siglip-so400m-patch14-384.json \
    ./configs/dataset/stair.json \
    ./configs/model/tanuki-moe.json \
    ./output_llava/checkpoints/pretrain-llava-jp-Tanuki-moe-vision-zero3-multinode-test \
    llava-jp-test \
    Tanuki-moe-vision-zero3-multinode-test
