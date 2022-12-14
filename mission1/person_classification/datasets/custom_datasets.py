import os
from os.path import join as opj
from glob import glob

import numpy as np
from PIL import Image

from .base_dataset import BaseDataset, is_img

class SelectiveDataset(BaseDataset):
    def __init__(self, data_root_dir, ds_names, cls_names, transform, sample_step=5, is_train=True, add_test_dataset=False):
        self.cls_names = cls_names
        self.transform = transform
        self.torv = "train" if is_train else "valid"
        self.add_test_dataset = add_test_dataset
        for label, cls_name in enumerate(cls_names):
            img_paths = []
            for ds in ds_names:
                _dir = opj(data_root_dir, ds, self.torv, cls_name)
                for root, _, fns in os.walk(_dir):
                    for fn in fns:
                        path = opj(root, fn)
                        if is_img(path):
                            img_paths.append(path)  
            img_paths = sorted(img_paths)
            labels = [label] * len(img_paths)
            setattr(self, f"{cls_name}_paths", img_paths[::sample_step])
            setattr(self, f"{cls_name}_labels", labels[::sample_step])
        if add_test_dataset:
            cls_dict = {"male": [0, 3], "female": [1,4], "kid" : [2,5]}
            cam_lst = sorted(os.listdir(opj(data_root_dir, "sample")))
            test_img_paths = []
            for label, cls_name in enumerate(cls_names):
                img_paths = []
                for cam in cam_lst:
                    for cls_ in cls_dict[cls_name]:
                        _dir = opj(data_root_dir, "sample", cam, str(cls_), "*")
                        img_paths.extend(glob(_dir))
                img_paths = sorted(img_paths)
                labels = [label] * len(img_paths)
                setattr(self, f"test_{cls_name}_paths", img_paths)
                setattr(self, f"test_{cls_name}_labels", labels)

        self.img_paths = []
        self.labels = []
        for cls_name in cls_names:
            self.img_paths.extend(getattr(self, f"{cls_name}_paths"))
            self.labels.extend(getattr(self, f"{cls_name}_labels"))
            if add_test_dataset:
                self.img_paths.extend(getattr(self, f"test_{cls_name}_paths"))
                self.labels.extend(getattr(self, f"test_{cls_name}_labels"))
    def check_paths(self):
        msg = ""
        for cls_name in self.cls_names:
            msg += f"[{self.torv}] cls_name : {cls_name}, # of images : {len(getattr(self, f'{cls_name}_paths'))}\n" 
            if self.add_test_dataset:
                msg += f"[Add test dataset - {self.torv}] cls_name : {cls_name}, # of images : {len(getattr(self, f'test_{cls_name}_paths'))}\n" 
        return msg
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.transform(image=np.array(Image.open(self.img_paths[idx]).convert("RGB")))["image"]
        label = self.labels[idx]
        return img, label
class TestDataset(BaseDataset):
    def __init__(self, data_root_dir, transform):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        cam_list = os.listdir(opj(data_root_dir, "sample"))
        for cam in cam_list:
            for i in range(6):
                paths = sorted(glob(opj(data_root_dir, "sample", cam, str(i), "*")))
                self.img_paths.extend(paths)
                self.labels.extend([i%3] * len(paths))
    def __len__(self):
        return len(self.img_paths)
    def check_paths(self):
        pass
    def __getitem__(self, idx):
        img = self.transform(image=np.array(Image.open(self.img_paths[idx]).convert("RGB")))["image"]
        label = self.labels[idx]
        return img, label
        
                

            