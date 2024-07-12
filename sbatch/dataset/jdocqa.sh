#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-19
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/prepare_dataset.out
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=24
#SBATCH --mem=200GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

python tools/pdf_to_images.py ./dataset/jdocqa/pdf/pdf_files ./dataset/jdocqa/images --num_processes 20
