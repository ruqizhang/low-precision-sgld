#!/bin/bash

WL=8
python -u train.py --dataset CIFAR10 \
       --model ResNet18LP \
       --epochs=245 \
       --lr_init=0.5 \
       --wd=5e-4 \
       --wl-weight ${WL} \
       --wl-grad ${WL} \
       --wl-activate ${WL} \
       --wl-error ${WL} \
       --fl-weight ${WL} \
       --seed 1 \
       --batch_size 128  \
       --weight-rounding stochastic \
       --grad-rounding stochastic \
       --activate-rounding stochastic \
       --error-rounding stochastic \
       --weight-type block \
       --grad-type block \
       --activate-type block \
       --error-type block \
       --noise 1 \
       --lr_type cyclic \
       --num_savemodel 35 \
       --quant_type vc;
