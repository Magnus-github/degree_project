import torch


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
