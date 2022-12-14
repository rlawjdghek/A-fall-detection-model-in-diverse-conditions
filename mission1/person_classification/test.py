import os
from os.path import join as opj

from utils.util import *
from main import test

#### config ####
load_dir = "/media/data1/jeonghokim/AGC_final/person_classification/20220930_resnet18basic"
config_path = opj(load_dir, "config.json")
model_save_path = opj(load_dir, "best.pth")
################
args = load_args(config_path)
test(args)

