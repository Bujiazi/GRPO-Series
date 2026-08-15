export NNODES=${NODE_COUNT:-1}
export PROC_PER_NODE=${PROC_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_PORT=29583

torchrun --nnodes=$NNODES --nproc_per_node=$PROC_PER_NODE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train.py \
    --method AdaGRPO \
    --seed 42 \
    --pretrained_model_name_or_path ./pretrained_models/FLUX \
    --prompt_txt_path ./data/prompts.txt \
    --output_dir ./outputs/adagrpo \
    --batch_size 10 \
    --dataloader_num_workers 4 \
    --mixed_precision "bf16" \
    --gradient_accumulation_steps 1 \
    --save_interval 100 \
    --height 512 \
    --width 512 \
    --train_mode "full" \
    --denoising_steps 16 \
    --sde_stepidx 0 2 4 6 \
    --sde_eta 0.7 \
    --reward_name "hpsv3" \
    --learning_rate 2e-6 \
    --group_size 12 \
    --max_grad_norm 1.0 \
    --max_train_steps 300 \
    --adv_clip_max 5.0 \
    --ema_alpha 0.8

