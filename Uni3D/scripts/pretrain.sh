#!/bin/bash
model=create_uni3d

clip_model="EVA02-E-14-plus" 
pretrained="laion2b_s9b_b144k"
embed_dim=1024

pc_model="eva02_base_patch14_448"
pretrained_pc="pretrained/model.safetensors"
pc_feat_dim=768

pc_encoder_dim=512

  torchrun --nnodes=1 \
    --nproc-per-node=2 \
    main.py \
    --model $model \
    --pretrain_dataset_name cap3d_ensemble \
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --clip-model $clip_model \
    --pc-model $pc_model \
    --pretrained $pretrained \
    --pretrained-pc $pretrained_pc \
    --warmup 10000 \
    --batch-size 48 \
    --epochs 200 \
    --pc-feat-dim=$pc_feat_dim \
    --pc-encoder-dim=$pc_encoder_dim \
    --embed-dim=$embed_dim \
    --lr=3e-4 \
    --point-lr=3e-4 \
    --drop-path-rate=0.20 \
    --wd=0.1 \
    --point-wd=0.1 \
    --ld=1.0 \
    --point-ld=0.95 \
    --grad-clip-norm=5.0 \
    --smoothing=0. \
    --seed 4096 \
    --patch-dropout=0.5 \
    --optimizer="adamw" \
    --enable-deepspeed \
    --zero-stage=1 \
    --openshape_setting \
    --validate_dataset_name modelnet40_openshape \
    --validate_dataset_name_lvis objaverse_lvis_openshape \
    --validate_dataset_name_scanobjnn scanobjnn_openshape \
    --wandb \
#    --resume /storage/combclip/2024_10_16-08_18_41-model_create_uni3d-lr_0.0003-b_48-j_4-p_amp/checkpoints
    # --use_lvis \ 
    # whether to use objaverse dataset during pretraining
    # whether to use objaverse dataset during pretraining
