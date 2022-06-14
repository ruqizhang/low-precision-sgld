#!/bin/bash

WL=8
python -u train.py --dataset CIFAR10 \
       --model ResNet18LP \
       --epochs=245 \
       --lr_init=0.5 \
       --wd=5e-4 \
       --wl-weight ${WL} \
       --wl-grad ${WL} \
       --fl-weight ${WL} \
       --fl-grad ${WL} \
       --seed 1 \
       --batch_size 128  \
       --weight-rounding stochastic \
       --grad-rounding stochastic \
       --weight-type fixed \
       --grad-type fixed\
       --noise 1 \
       --quant_acc -2;
