import torch

from utils.str_to_class import str_to_class

class Trainer:
    def __init__(self, cfg, dataloaders):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cfg = cfg
        self.model = str_to_class(cfg.model.name)(**cfg.model.params).to(self.device)
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.optimizer = str_to_class(cfg.hparams.optimizer.name)(self.model.parameters(), **cfg.hparams.optimizer.params)
        self.scheduler = str_to_class(cfg.hparams.scheduler.name)(self.optimizer, **cfg.hparams.scheduler.params)
        self.criterion = str_to_class(cfg.hparams.criterion.name)(**cfg.hparams.criterion.params)