#!/bin/sh
python /ailab/user/fanyimin/zhongyunhua/MultiGPS/main.py \
    --adata_file /ailab/user/fanyimin/zhongyunhua/MultiGPS/scripts/ALM_preprocessed.h5ad \
    --task_name ALM_MultiGPS \
    --seed 1 \
    --tasks recon,cls \
    --hvg \
    --saving_dir /ailab/user/fanyimin/zhongyunhua/MultiGPS/test/ALM/index/ \
    --log_dir /ailab/user/fanyimin/zhongyunhua/MultiGPS/test/ALM/log/ \
    --device cuda:0 \
    --epoch 2000 \
    --layer log1pcpm \
    --val \
    --record 100 \
    --activation tanh \
    --batch_size 1024 \
    --learning_rate 0.001