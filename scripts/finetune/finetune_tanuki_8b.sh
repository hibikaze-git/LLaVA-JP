#!/bin/bash

python train_llava.py \
    --base_model llama \
    --model_name_or_path hatakeyama-llm-team/Tanuki-8B \
    --version v1 \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_layer -2 \
    --pretrain_mm_mlp_adapter ./output_llava/checkpoints/pretrain-llava-jp-1.3b-v1-siglip-so400m-patch14-384/checkpoint-2000/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_feature patch \
    --data_path ./dataset/llava_visual_genome_ja.json \
    --lazy_preprocess False \
    --is_multimodal True \
    --image_folder ./dataset/images/stage2 \
    --image_aspect_ratio square \
    --optim adamw_bnb_8bit \
    --double_quant True \
    --quant_type nf4 \
    --bits 16 \
    --lora_enable False \
    --group_by_modality_length True \
    --fp16 False \
    --bf16 True \
    --output_dir ./output_llava/checkpoints/finetune-llava-jp-Tanuki-8B-v1-siglip-so400m-patch14-384 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 1532 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine" \
    --use_wandb \
    --wandb_project llava-jp-finetune-Tanuki-8B-test \
    --wandb_name Tanuki-8B