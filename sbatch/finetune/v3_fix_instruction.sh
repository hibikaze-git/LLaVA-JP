#!/bin/bash

# Command line options go here
#SBATCH --time=24:45:00
#SBATCH --nodelist=slurm0-a3-ghpc-6
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/train_llava-jp_stage2_fix_instruction.out
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=150
#SBATCH --mem=1000GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

bash scripts/finetune/finetune_accelerate.sh \
    ./configs/train/finetune/batch_128.json \
    ./configs/image_encoder/siglip-so400m-patch14-384.json \
    ./configs/dataset/v3_stage_2_fix_instruction.json \
    ./configs/model/tanuki-8b.json \
    ./output_llava/checkpoints/finetune-llava-jp-Tanuki-8B-vision-v3_fix_instruction \
    llava-jp-stage2 \
    Tanuki-8B-vision-v3_fix_instruction \
    ./output_llava/checkpoints/pretrain-llava-jp-Tanuki-8B-vision-v1/mm_projector.bin
