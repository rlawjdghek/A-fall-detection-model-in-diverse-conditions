#! /bin/bash
# bash run_models.sh <try>    e.x) bash run_models.sh size

CUDA_VISIBLE_DEVICES=0 nohup python main.py --m1_person_model_name resnet18 --save_name resnet18_$1 &> ./exp_nohup/resnet18_$1.txt &
CUDA_VISIBLE_DEVICES=0 nohup python main.py --m1_person_model_name tf_efficientnet_b1_ns --save_name tf_efficientnet_b1_ns_$1 &> ./exp_nohup/tf_efficientnet_b1_ns_$1.txt &
CUDA_VISIBLE_DEVICES=1 nohup python main.py --m1_person_model_name resnet50 --save_name resnet50_$1 &> ./exp_nohup/resnet50_$1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python main.py --m1_person_model_name resnet101 --save_name resnet101_$1 &> ./exp_nohup/resnet101_$1.txt & 
CUDA_VISIBLE_DEVICES=3 nohup python main.py --m1_person_model_name tf_efficientnet_b0_ns --save_name tf_efficientnet_b0_ns_$1 &> ./exp_nohup/tf_efficientnet_b0_ns_$1.txt &
CUDA_VISIBLE_DEVICES=4 nohup python main.py --m1_person_model_name gluon_resnet50_v1b --save_name gluon_resnet50_v1b_$1 &> ./exp_nohup/gluon_resnet50_v1b_$1.txt &
CUDA_VISIBLE_DEVICES=5 nohup python main.py --m1_person_model_name tf_mobilenetv3_small_075 --save_name tf_mobilenetv3_small_075_$1 &> ./exp_nohup/tf_mobilenetv3_small_075_$1.txt &
CUDA_VISIBLE_DEVICES=5 nohup python main.py --m1_person_model_name tf_efficientnet_b2_ns --save_name tf_efficientnet_b2_ns_$1 &> ./exp_nohup/tf_efficientnet_b2_ns_$1.txt &
CUDA_VISIBLE_DEVICES=6 nohup python main.py --m1_person_model_name vgg11 --save_name vgg11_$1 &> ./exp_nohup/vgg11_$1.txt &
CUDA_VISIBLE_DEVICES=7 nohup python main.py --m1_person_model_name resnext26ts --save_name resnext26ts_$1 &> ./exp_nohup/resnext26ts_$1.txt &
