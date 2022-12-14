from abc import ABC, abstractmethod
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMG_SUF = [".png", ".PNG", ".jpg", ".JPG", ".JPEG"]
def is_img(path):
    _, ext = os.path.splitext(path)
    return ext in IMG_SUF
def get_transform(args, is_train=True):
    if is_train:
        transform = A.Compose( 
            A.Normalize(),
            A.Resize(height=args.img_H, width=args.img_W),
            ToTensorV2()
        )
    else:
        transform = A.Compose( 
            A.Normalize(),
            A.Resize(height=args.img_H, width=args.img_W),
            ToTensorV2()
        )
    return transform
class BaseDataset(Dataset, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
    @abstractmethod
    def check_paths(self):
        pass