#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --job-name=dl_data
#SBATCH --output=sbatch_logs/download_commoncatalog-cc-by_photoid.out
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

python tools/commoncatalog-cc-by-recap-diverse_questions/download_images.py
