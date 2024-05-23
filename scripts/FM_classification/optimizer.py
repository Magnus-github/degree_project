import torch
import math


class SGD(torch.optim.SGD):
    def __init__(self, params, lr=0, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

class Adam(torch.optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super(Adam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

class AdamW(torch.optim.AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
        super(AdamW, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

class CyclicLR(torch.optim.lr_scheduler.CyclicLR):
    def __init__(self, optimizer, base_lr=0.001, max_lr=0.006, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
        super(CyclicLR, self).__init__(optimizer, base_lr, max_lr, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode, cycle_momentum, base_momentum, max_momentum, last_epoch)

class StepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super(StepLR, self).__init__(optimizer, step_size, gamma, last_epoch)

class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        super(ReduceLROnPlateau, self).__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps)


class CosineDecayWithWarmUpScheduler(object):
    def __init__(self,optimizer,step_per_epoch,init_warmup_lr=1e-5,warm_up_steps=1000,max_lr=1e-4,min_lr=1e-6,num_step_down=2000,num_step_up=None,
                T_mul=1,max_lr_decay=None, gamma=1,min_lr_decay=None,alpha=1):
        self.optimizer = optimizer
        self.step_per_epoch = step_per_epoch
        if warm_up_steps != 0:
            self.warm_up = True
        else:
            self.warm_up = False  
        self.init_warmup_lr = init_warmup_lr
        self.warm_up_steps = warm_up_steps
        self.max_lr = max_lr
        if min_lr == 0:
            self.min_lr = 0.1 * max_lr
            self.alpha = 0.1
        else:
            self.min_lr = min_lr
        self.num_step_down = num_step_down
        if num_step_up == None:
            self.num_step_up = num_step_down
        else:
            self.num_step_up = num_step_up    
        self.T_mul = T_mul
        if max_lr_decay == None:
            self.gamma = 1
        elif max_lr_decay == 'Half':
            self.gamma = 0.5
        elif max_lr_decay == 'Exp':
            self.gamma = gamma
        
        if min_lr_decay == None:
            self.alpha = 1
        elif min_lr_decay == 'Half':
            self.alpha = 0.5
        elif min_lr_decay == 'Exp':
            self.alpha = alpha


        self.num_T = 0
        self.iters = 0
        self.lr_list = []
        
        
    def update_cycle(self, lr):
        old_min_lr = self.min_lr
        if lr == self.max_lr or (self.num_step_up == 0 and lr == self.min_lr):
            if self.num_T == 0:
                self.warm_up = False
                self.min_lr /= self.alpha
            self.iters = 0
            self.num_T += 1
            self.min_lr *= self.alpha

        if lr == old_min_lr and self.max_lr * self.gamma >= self.min_lr:
            self.max_lr *= self.gamma
            
    
    def step(self):
        self.iters += 1
        if self.warm_up:
            lr = self.init_warmup_lr + (self.max_lr-self.init_warmup_lr) / self.warm_up_steps * self.iters
        else:
            T_cur = self.T_mul**self.num_T
            if self.iters <= self.num_step_down*T_cur:
                lr = self.min_lr + (self.max_lr-self.min_lr) * (1 + math.cos(math.pi*self.iters/(self.num_step_down*T_cur)))/2
                if lr < self.min_lr:
                    lr = self.min_lr
            elif self.iters > self.num_step_down*T_cur:
                lr = self.min_lr + (self.max_lr-self.min_lr) / (self.num_step_up * T_cur) * (self.iters-self.num_step_down*T_cur)
                if lr > self.max_lr:
                    lr = self.max_lr

        self.update_cycle(lr)
                
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            self.lr_list.append(lr)