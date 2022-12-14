#! /bin/bash

tail -n 15 ./exp_nohup/gluon_resnet50_v1b_$1.txt 
tail -n 15 ./exp_nohup/resnet101_$1.txt 
tail -n 15 ./exp_nohup/resnet18_$1.txt 
tail -n 15 ./exp_nohup/resnet50_$1.txt 
tail -n 15 ./exp_nohup/resnext26ts_$1.txt 
tail -n 15 ./exp_nohup/tf_efficientnet_b0_ns_$1.txt 
tail -n 15 ./exp_nohup/tf_efficientnet_b1_ns_$1.txt 
tail -n 15 ./exp_nohup/tf_efficientnet_b2_ns_$1.txt 
tail -n 15 ./exp_nohup/tf_mobilenetv3_small_075_$1.txt 
tail -n 15 ./exp_nohup/vgg11_$1.txt 

