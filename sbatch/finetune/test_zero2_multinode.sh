#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-[6,17]
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/train_llava-jp_stage2.out
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=150
#SBATCH --mem=1500GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp-multinode

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

bash scripts/finetune/finetune_accelerate_zero2_multinode.sh \
    ./configs/train/finetune/batch_per_device_1.json \
    ./configs/image_encoder/siglip-so400m-patch14-384.json \
    ./configs/dataset/ja-vg-vqa.json \
    ./configs/model/tanuki-8b.json \
    ./output_llava/checkpoints/finetune-llava-jp-Tanuki-8B-vision-zero2-multinode-test \
    llava-jp-test \
    Tanuki-8B-vision-zero2-multinode-test \
    ./output_llava/checkpoints/pretrain-llava-jp-Tanuki-8B-vision-zero2-multinode-test/mm_projector.bin
