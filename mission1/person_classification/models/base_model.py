from abc import ABC, abstractmethod

import torch
from torch.optim import lr_scheduler
from torch import optim
def define_optimizer(args, model):
    if args.optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError(args.optim_name)
    return optimizer
def define_scheduler(args, optimizer):
    if args.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=0.1, last_epoch=-1)
    elif args.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    else:
        raise NotImplementedError(args.lr_policy)
    return scheduler
def define_criterion(args):
    if args.loss_name == "ce":
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        raise NotImplementedError(args.loss_name)
    return criterion
class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
    @abstractmethod
    def set_input(self): pass
    @abstractmethod
    def train(self): pass
    @staticmethod
    def set_requires_grad(models, requires_grad=False):
        if not isinstance(models, list):
            models = [models]
        for model in models:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = requires_grad
