#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-19
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/train_llava-jp_stage2.out
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=500GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

bash scripts/finetune/finetune_accelerate.sh \
    ./configs/train/finetune/batch_16-epoch_3.json \
    ./configs/image_encoder/siglip-base-patch16-256-multilingual.json \
    ./configs/dataset/jdocqa_only_answerable.json \
    ./configs/model/Tanuki-8B-vision-cc300k_j-vg-vqa-s2_siglip_256.json \
    ./output_llava/checkpoints/finetune-llava-jp-Tanuki-8B-vision-cc300k_j-vg-vqa-s2_siglip_256-jdocqa-batch16 \
    llava-jp-stage2 \
    Tanuki-8B-vision-cc300k_j-vg-vqa-s2_siglip_256-jdocqa-batch16 \
    ./output_llava/checkpoints/pretrain-llava-jp-Tanuki-8B-vision-cc300k-s2_siglip_256/mm_projector.bin
