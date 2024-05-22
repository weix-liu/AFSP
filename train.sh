#!/bin/sh
#CUDA_VISIBLE_DEVICES=$1 python  trainval_SF_source.py   --cuda --lr 0.001  --net res101  --dataset gta_car --dataset_t  ucas_car   --save_dir training/SF
CUDA_VISIBLE_DEVICES=$1 python trainval_SF_mt_afsp.py --cuda --lr 0.001  --net res101  --dataset gta_car --dataset_t  ucas_car   --save_dir training/SF --load_name training/SF/res101/gta_car/SF_source_False_target_ucas_car_gamma_5_1_3_9999.pth
