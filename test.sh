#!/bin/sh
#CUDA_VISIBLE_DEVICES=$1 python  test_SF_source.py  --dataset ucas_car --net res101 --cuda --load_name training/SF/res101/gta_car/SF_source_False_target_ucas_car_gamma_5_1_3_9999.pth
CUDA_VISIBLE_DEVICES=$1 python test_SF_mt_afsp.py --dataset ucas_car --net res101 --cuda --load_name training/SF/res101/gta_car/SF_mt_afsp_target_ucas_car_1_7_1019.pth