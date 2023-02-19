#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4

python -m torch.distributed.launch \
    --master_port 12324 \
    main.py \
    --distributed --multiGPU \
        --task beauty \
        --seed 2022 \
        --warmup_prop 0.05 \
        --lr 1e-3 \
        --clip 1.0 \
        --model_type 't5-small' \
        --epoch 10 \
        --gpu '1,2,3,4' \
        --logging_step 1000 \
        --logging_dir 'pretrain_t5_small_beauty.log' \
        --model_dir 'pretrain_t5_small_beauty.pt' \
        #--item_representation random_one_token
