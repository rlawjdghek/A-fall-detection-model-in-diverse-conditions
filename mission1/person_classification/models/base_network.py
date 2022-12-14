import timm
import torch
import torch.nn as nn

def he_init(module):
    if isinstance(module, nn.Conv2d): 
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
def define_model(args):
    model = timm.create_model(model_name=args.m1_person_model_name, pretrained=not args.m1_person_no_pretrained, num_classes=len(args.m1_person_cls_names))
    if args.m1_person_no_pretrained:
        model.apply(he_init)
    return model
def count_params(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    return n_params
def get_lr(optimizer):
    for g in optimizer.param_groups:
        return g["lr"]
def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)
