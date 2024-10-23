#!/bin/sh
python /ailab/user/fanyimin/zhongyunhua/MultiGPS/eval.py \
--file_name ALM_MultiGPS \
--gene_panel /ailab/user/fanyimin/zhongyunhua/MultiGPS/test/ALM/index/recon-cls-seed1-ALM_MultiGPS-epoch:999.txt \
--gene_number 64 \
--target_adata /ailab/user/fanyimin/zhongyunhua/MultiGPS/scripts/ALM_preprocessed.h5ad \
--output_dir /ailab/user/fanyimin/zhongyunhua/MultiGPS/test/ALM/eval/ \
--seed 1 \
--device cuda:0
