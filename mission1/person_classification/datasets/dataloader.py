import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .custom_datasets import SelectiveDataset, TestDataset
def get_transform(args, is_train=True):
    if is_train:
        # transform = A.Compose([  # TODO : 매개변수화 시키기
        #     A.Resize(height=args.img_H, width=args.img_W),
        #     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        #     A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        #     A.RandomBrightnessContrast(p=0.5),
        #     A.ColorJitter(),
        #     A.Normalize(),
        #     ToTensorV2()
        # ])
        transform = A.Compose([
            A.RandomResizedCrop(height=args.crop_H, width=args.crop_W),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(height=args.crop_H, width=args.crop_W),
            A.Normalize(),
            ToTensorV2()
        ])
    return transform
def get_dataloader(args):
    train_transform = get_transform(args, is_train=True)
    valid_transform = get_transform(args, is_train=False)
    train_dataset = SelectiveDataset(args.data_root_dir, ds_names=args.train_ds_names, cls_names=args.cls_names, transform=train_transform, sample_step=args.sample_step, is_train=True, add_test_dataset=args.add_test_dataset)
    valid_dataset = SelectiveDataset(args.data_root_dir, ds_names=args.valid_ds_names, cls_names=args.cls_names, transform=valid_transform, sample_step=args.val_sample_step, is_train=False)
    if args.use_DDP:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=train_sampler, num_workers=args.n_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    return train_loader, valid_loader
def get_testloader(args):
    test_transform = get_transform(args, is_train=False)
    test_dataset = TestDataset(args.data_root_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    return test_loader