import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_model import BaseModel, define_optimizer, define_scheduler, define_criterion
from .base_network import define_model, count_params, get_lr, moving_average


class PersonClassification(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = define_model(args).cuda(args.local_rank)
        self.model_ema = define_model(args).cuda(args.local_rank).eval()        
        self.n_params_model = count_params(self.model)
        if not args.is_test:
            if args.use_DDP:
                self.model = DDP(self.model, device_ids=[args.local_rank])
            self.optimizer = define_optimizer(args, self.model)
            self.scheduler = define_scheduler(args, self.optimizer)
            self.criterion = define_criterion(args)
    def set_input(self, img, label):
        self.img = img
        self.label = label
    def train(self):
        output = self.model(self.img)
        loss = self.criterion(output, self.label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, output.detach()
    def scheduler_step(self):
        if self.args.lr_policy == "step":
            self.scheduler.step()
        elif self.args.lr_policy == "cosine":
            self.scheduler.step()
        else:
            NotImplementedError(self.args.lr_policy)
    def get_learning_rate(self):
        lr = get_lr(self.optimizer)
        return lr
    @torch.no_grad()
    def inference(self, img):
        output = self.model(img)
        return output
    def ema_inference(self, img):
        output = self.model_ema(img)
        return output
    def to_eval(self):
        self.model.eval()
        self.model_ema.eval()
    def to_train(self):
        self.model.train()
    def load(self, load_path):
        state_dict = torch.load(load_path)
        self.model.load_state_dict(state_dict["model"])
        self.model_ema.load_state_dict(state_dict["model_ema"])
    def save(self, save_path):
        state_dict = {}
        if self.args.use_DDP:
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()
        state_dict["model_ema"] = self.model_ema.state_dict()
        if self.args.local_rank == 0:
            torch.save(state_dict, save_path)
    def update_moving_avg(self):
        if self.args.use_DDP:   
            moving_average(self.model.module, self.model_ema, beta=self.args.ema_beta)         
        else:
            moving_average(self.model, self.model_ema, beta=self.args.ema_beta)         
            
    
        
        
        
