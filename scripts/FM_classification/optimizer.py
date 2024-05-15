import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi, floor, log
from typing import Iterable


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


class CyclicCosineDecayLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 init_decay_epochs,
                 min_decay_lr,
                 restart_interval=None,
                 restart_interval_multiplier=None,
                 restart_lr=None,
                 warmup_epochs=None,
                 warmup_start_lr=None,
                 last_epoch=-1):
        """
        Initialize new CyclicCosineDecayLR object.

        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_decay_epochs: (int) - Number of initial decay epochs.
        :param min_decay_lr: (float or iterable of floats) - Learning rate at the end of decay.
        :param restart_interval: (int) - Restart interval for fixed cycles.
            Set to None to disable cycles. Default: None.
        :param restart_interval_multiplier: (float) - Multiplication coefficient for geometrically increasing cycles.
            Default: None.
        :param restart_lr: (float or iterable of floats) - Learning rate when cycle restarts.
            If None, optimizer's learning rate will be used. Default: None.
        :param warmup_epochs: (int) - Number of warmup epochs. Set to None to disable warmup. Default: None.
        :param warmup_start_lr: (float or iterable of floats) - Learning rate at the beginning of warmup.
            Must be set if warmup_epochs is not None. Default: None.
        :param last_epoch: (int) - The index of the last epoch. This parameter is used when resuming a training job. Default: -1.
        """

        if not isinstance(init_decay_epochs, int) or init_decay_epochs < 1:
            raise ValueError("init_decay_epochs must be positive integer, got {} instead".format(init_decay_epochs))

        if isinstance(min_decay_lr, Iterable) and len(min_decay_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(min_decay_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(min_decay_lr), len(optimizer.param_groups)))

        if restart_interval is not None and (not isinstance(restart_interval, int) or restart_interval < 1):
            raise ValueError("restart_interval must be positive integer, got {} instead".format(restart_interval))

        if restart_interval_multiplier is not None and \
                (not isinstance(restart_interval_multiplier, float) or restart_interval_multiplier <= 0):
            raise ValueError("restart_interval_multiplier must be positive float, got {} instead".format(
                restart_interval_multiplier))

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(restart_lr), len(optimizer.param_groups)))

        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
                raise ValueError(
                    "Expected warmup_epochs to be positive integer, got {} instead".format(type(warmup_epochs)))

            if warmup_start_lr is None:
                raise ValueError("warmup_start_lr must be set when warmup_epochs is not None")

            if not (isinstance(warmup_start_lr, float) or isinstance(warmup_start_lr, Iterable)):
                raise ValueError("warmup_start_lr must be either float or iterable of floats, got {} instead".format(
                    warmup_start_lr))

            if isinstance(warmup_start_lr, Iterable) and len(warmup_start_lr) != len(optimizer.param_groups):
                raise ValueError("Expected len(warmup_start_lr) to be equal to len(optimizer.param_groups), "
                                 "got {} and {} instead".format(len(warmup_start_lr), len(optimizer.param_groups)))

        group_num = len(optimizer.param_groups)
        self._warmup_start_lr = [warmup_start_lr] * group_num if isinstance(warmup_start_lr, float) else warmup_start_lr
        self._warmup_epochs = 0 if warmup_epochs is None else warmup_epochs
        self._init_decay_epochs = init_decay_epochs
        self._min_decay_lr = [min_decay_lr] * group_num if isinstance(min_decay_lr, float) else min_decay_lr
        self._restart_lr = [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        self._restart_interval = restart_interval
        self._restart_interval_multiplier = restart_interval_multiplier
        super(CyclicCosineDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return self._calc(self.last_epoch,
                              self._warmup_epochs,
                              self._warmup_start_lr,
                              self.base_lrs)

        elif self.last_epoch < self._init_decay_epochs + self._warmup_epochs:
            return self._calc(self.last_epoch - self._warmup_epochs,
                              self._init_decay_epochs,
                              self.base_lrs,
                              self._min_decay_lr)
        else:
            if self._restart_interval is not None:
                if self._restart_interval_multiplier is None:
                    cycle_epoch = (self.last_epoch - self._init_decay_epochs - self._warmup_epochs) % self._restart_interval
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      self._restart_interval,
                                      lrs,
                                      self._min_decay_lr)
                else:
                    n = self._get_n(self.last_epoch - self._warmup_epochs - self._init_decay_epochs)
                    sn_prev = self._partial_sum(n)
                    cycle_epoch = self.last_epoch - sn_prev - self._warmup_epochs - self._init_decay_epochs
                    interval = self._restart_interval * self._restart_interval_multiplier ** n
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      interval,
                                      lrs,
                                      self._min_decay_lr)
            else:
                return self._min_decay_lr

    def _calc(self, t, T, lrs, min_lrs):
        return [min_lr + (lr - min_lr) * ((1 + cos(pi * t / T)) / 2)
                for lr, min_lr in zip(lrs, min_lrs)]

    def _get_n(self, epoch):
        _t = 1 - (1 - self._restart_interval_multiplier) * epoch / self._restart_interval
        return floor(log(_t, self._restart_interval_multiplier))

    def _partial_sum(self, n):
        return self._restart_interval * (1 - self._restart_interval_multiplier ** n) / (
                    1 - self._restart_interval_multiplier)