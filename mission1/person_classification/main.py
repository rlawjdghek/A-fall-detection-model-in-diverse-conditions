import os
from os.path import join as opj
import argparse
import datetime
import time
import shutil

import torch
import torch.distributed as dist
from torch.backends import cudnn

from person_utils.util import *
from datasets.dataloader import get_dataloader, get_testloader

from models.pc import PersonClassification
def build_args(is_test=False):
    parser = argparse.ArgumentParser()

    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/data/AGC_final/person_classification")
    parser.add_argument("--train_ds_names", default=["AIHUB1", "AIHUB2", "PA-100K"])
    parser.add_argument("--valid_ds_names", default=["AIHUB1", "AIHUB2", "PA-100K"])
    parser.add_argument("--m1_person_cls_names", default=["male", "female", "kid"])
    parser.add_argument("--img_H", type=int, default=256)
    parser.add_argument("--img_W", type=int, default=144)
    parser.add_argument("--crop_H", type=int, default=224)
    parser.add_argument("--crop_W", type=int, default=128)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--sample_step", type=int, default=1)
    parser.add_argument("--val_sample_step", type=int, default=5)
    parser.add_argument("--add_test_dataset", type=bool, default=True)

    #### model ####
    parser.add_argument("--m1_person_model_name", type=str, default="resnet18")
    parser.add_argument("--m1_person_no_pretrained", action="store_true")
    
    #### train & eval ####
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr_decay_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--optim_name", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("--lr_policy", type=str, default="step", choices=["step", "cosine"])
    parser.add_argument("--loss_name", type=str, default="ce", choices=["ce"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=int, default=1e-5)
    parser.add_argument("--betas", default=[0.9, 0.999])
    parser.add_argument("--ema_beta", type=float, default=0.999)
    #### save & load ####
    parser.add_argument("--no_save", type=bool, default=False)
    parser.add_argument("--save_root_dir", type=str, default="/media/data1/jeonghokim/AGC_final/person_classification")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--log_save_iter_freq", type=int, default=100)

    #### config ####
    parser.add_argument("--use_DDP", action="store_true")    

    args = parser.parse_args()
    args.save_name = f"{datetime.datetime.now().strftime('%Y%m%d')}_" + args.save_name
    args.is_test = is_test
    if is_test:
        args.use_DDP = False
        args.no_save = True
    if args.use_DDP:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=72000))
        args.n_gpus = dist.get_world_size()
    else:
        args.local_rank = 0
        args.n_gpus = 1
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.img_save_dir = opj(args.save_dir, "save_images")
    args.model_save_dir = opj(args.save_dir, "save_models")
    args.eval_save_dir = opj(args.save_dir, "eval_save_images")
    args.log_path = opj(args.save_dir, "log.txt")
    args.config_path = opj(args.save_dir, "config.json")
    if not args.no_save:
        os.makedirs(args.img_save_dir, exist_ok=True)
        os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.eval_save_dir, exist_ok=True)
    return args

def main_worker(args, logger):
    if args.local_rank == 0:
        save_args(args, args.config_path)
    train_loader, valid_loader = get_dataloader(args)
    logger.write(train_loader.dataset.check_paths())
    logger.write(valid_loader.dataset.check_paths())
    model = PersonClassification(args)
    train_acc_meter = AverageMeter()
    valid_acc_meter = AverageMeter()
    valid_acc_ema_meter = AverageMeter()
    train_loss_meter = AverageMeter()
    best_valid_acc = -1
    start_time = time.time()
    cur_iter = 1
    for epoch in range(1, args.n_epochs+1):
        if args.use_DDP:
            train_loader.sampler.set_epoch(epoch)
        model.to_train()
        for i, (img, label) in enumerate(train_loader):
            BS = img.shape[0]
            img = img.cuda(args.local_rank)
            label = label.cuda(args.local_rank)
            model.set_input(img, label)
            loss, output = model.train()

            acc1 = Accuracy(output, label, topk=(1,))[0]
            model.update_moving_avg()
            train_loss_meter.update(loss.item(), BS)
            train_acc_meter.update(acc1.item(), BS)         
            if cur_iter % args.log_save_iter_freq == 0:
                elapsed_time = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
                msg = f"[Train]_[Elapsed Time - {elapsed_time}]_[epoch - {epoch}]_[iter - {i}/{len(train_loader)}]_[loss - {train_loss_meter.avg:.2f}]"
                logger.write(msg)
            cur_iter += 1
        lr = model.get_learning_rate()
        logger.write("="*30)
        msg = f"[Train]_[epoch - {epoch}]_[Top1 Acc - {train_acc_meter.avg:.2f}]_[lr - {lr}]"
        logger.write(msg)

        # validation
        if args.local_rank == 0:
            model.to_eval()
            pred_lst = []
            pred_ema_lst = []
            label_lst = []
            with torch.no_grad():
                for i, (img, label) in enumerate(valid_loader):
                    BS = img.shape[0]
                    img = img.cuda(args.local_rank)
                    label = label.cuda(args.local_rank)
                    output = model.inference(img)
                    output_ema = model.ema_inference(img)

                    acc1 = Accuracy(output, label, topk=(1,))[0]
                    acc1_ema = Accuracy(output_ema, label, topk=(1, ))[0]

                    pred = output.argmax(dim=1).detach().cpu().tolist()
                    pred_ema = output_ema.argmax(dim=1).detach().cpu().tolist()
                    pred_lst.extend(pred)
                    pred_ema_lst.extend(pred_ema)
                    label_lst.extend(label.detach().cpu().tolist())
                    
                    valid_acc_meter.update(acc1.item(), BS)
                    valid_acc_ema_meter.update(acc1_ema.item(), BS)
            valid_acc_val = valid_acc_meter.avg
            valid_acc_ema_val = valid_acc_ema_meter.avg
            msg = f"[Valid]_[epoch - {epoch}]_[Top1 Acc - {valid_acc_val:.2f}]_[Top1 Acc ema - {valid_acc_ema_val:.2f}]"
            logger.write(msg)        
            cm = get_confusion_mat(pred_lst, label_lst)
            cm_ema = get_confusion_mat(pred_ema_lst, label_lst)
            logger.write(f"{cm}")
            logger.write(f"{cm_ema}")
            logger.write("="*30)
            
            if best_valid_acc < valid_acc_val:
                best_valid_acc = valid_acc_val
                save_path = opj(args.model_save_dir, f"[Epoch-{epoch}]_[acc-{valid_acc_val:.2f}].pth")
                model.save(save_path)
                best_save_path = opj(args.model_save_dir, "best.pth")
                shutil.copy(save_path, best_save_path)
            
        if args.use_DDP:
            dist.barrier()
        model.scheduler_step()
        
    last_save_path = opj(args.model_save_dir, "last.pth")
    model.save(last_save_path)    
    
def test(args, logger=None):
    test_loader = get_testloader(args)
    test_acc_meter = AverageMeter()
    test_acc_ema_meter = AverageMeter()
    model = PersonClassification(args)
    load_path = opj(args.model_save_dir, "best.pth")
    model.load(load_path)
    model.to_eval()
    pred_lst = []
    pred_ema_lst = []
    label_lst = []
    with torch.no_grad():
        for img, label in test_loader:
            BS = img.shape[0]
            img = img.cuda(args.local_rank)
            label = label.cuda(args.local_rank)
            output = model.inference(img)
            output_ema = model.ema_inference(img)
            pred = output.argmax(dim=1).detach().cpu().tolist()
            pred_ema = output_ema.argmax(dim=1).detach().cpu().tolist()
            pred_lst.extend(pred)
            pred_ema_lst.extend(pred_ema)
            label_lst.extend(label.detach().cpu().tolist())
            acc1 = Accuracy(output, label, topk=(1,))[0]
            acc1_ema = Accuracy(output_ema, label, topk=(1,))[0]

            test_acc_meter.update(acc1.item(), BS)
            test_acc_ema_meter.update(acc1_ema.item(), BS)
        test_acc_val = test_acc_meter.avg
        test_acc_ema_val = test_acc_ema_meter.avg
    msg = f"[Test]_[Top1 Acc - {test_acc_val:.2f}]_[Top1 Acc ema - {test_acc_ema_val:.2f}]"
    cm = get_confusion_mat(pred_lst, label_lst)
    cm_ema = get_confusion_mat(pred_ema_lst, label_lst)
    print("="*30)
    print("="*30)
    if logger:
        logger.write(msg)
        logger.write(f"{cm}")
        logger.write(f"{cm_ema}")
    else:
        print(msg)
        print(cm)
        print(cm_ema)
                
if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.log_path)
    print_args(args, logger)
    cudnn.benchmark = True
    main_worker(args, logger)
    if args.local_rank == 0:
        args.is_test = True
        args.use_DDP = False
        test(args, logger)
    if args.use_DDP:
        dist.barrier()

