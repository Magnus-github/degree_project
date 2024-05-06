import torch
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import logging

import sys
import os
sys.path.append(".")
from data.dataloaders import get_dataloaders
from scripts.utils.str_to_class import str_to_class
from scripts.utils.check_debugger import debugger_is_active

class Trainer:
    def __init__(self, cfg, dataloaders, run_id=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cfg = cfg
        self.output_dir = os.path.join(self._cfg.save_path, str(run_id))
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = str_to_class(cfg.model.name)(**cfg.model.params).to(self.device)
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.optimizer = str_to_class(cfg.hparams.optimizer.name)(self.model.parameters(), **cfg.hparams.optimizer.params)
        self.scheduler = str_to_class(cfg.hparams.scheduler.name)(self.optimizer, **cfg.hparams.scheduler.params)
        # self.criterion = str_to_class(cfg.hparams.criterion.name)(**cfg.hparams.criterion.params)
        self.criterion = self.model.loss_function

        if not debugger_is_active() and cfg.wandb.enable:
            wandb.watch(self.model)
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self):
        self.model.train()
        for epoch in range(self._cfg.hparams.epochs):
            running_loss = 0.0
            last_val_losses = []
            for i, data in enumerate(tqdm(self.train_loader)):
                inputs = data[0]
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs['pred'], inputs, outputs['distribution'])['loss']
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if not debugger_is_active() and self._cfg.wandb.enable:
                    last_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({"Train Loss [batch]": loss.item(), "Learning Rate": last_lr})

                if i > 1:
                    break
                
            self.logger.info(f"Epoch {epoch} Loss: {running_loss / len(self.train_loader)}")
            val_loss = self.validate()
            
            if not debugger_is_active() and self._cfg.wandb.enable:
                wandb.log({"Train Loss [epoch]": running_loss / len(self.train_loader), "Val Loss": val_loss, "Epoch": epoch})

            last_val_losses.append(val_loss)
            if len(last_val_losses) > 5:
                if last_val_losses[-1] > last_val_losses[-2] > last_val_losses[-3] > last_val_losses[-4] > last_val_losses[-5]:
                    self.logger.info("Early stopping")
                    break

        torch.save(self.model.state_dict(), self.output_dir+"/model.pth")

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            for i, data in enumerate(tqdm(self.val_loader)):
                inputs = data[0]
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs['pred'], inputs, outputs['distribution'])['loss']
                running_val_loss += loss.item()

                if i == 0:
                    self.plot_reconstruction(inputs, outputs['pred'])

        self.logger.info(f"Validation Loss: {running_val_loss / len(self.val_loader)}")

        return running_val_loss / len(self.val_loader)
    
    def plot_reconstruction(self, inputs, pred):
        input = inputs.cpu()
        input = input.reshape(-1, 4, 18, 5)
        input = input.permute(0, 3, 1, 2).numpy()
        pred = pred.cpu()
        pred = pred.reshape(-1, 4, 18, 5)
        pred = pred.permute(0, 3, 1, 2).numpy()
        fig, ax = plt.subplots(18, 4, figsize=(15, 5*18))
        for i in range(0,501,100):
            for j in range(4):
                for k in range(18):
                    ax[k, j].plot(input[i,:,j,k], label="GT")
                    ax[k, j].plot(pred[i,:,j,k], label="Pred")
                    ax[k, j].set_title(f"Joint {k} - {['x', 'y', 'v_x', 'v_y'][j]}")
                    ax[k, j].legend()
            plt.savefig(f"reconstruction_{i}.png")
            plt.close()
            






def main():
    # Load the configuration file
    cfg = OmegaConf.load("config/train_AE.yaml")
    dataloaders = get_dataloaders(cfg)
    trainer = Trainer(cfg, dataloaders)
    trainer.train()


if __name__ == "__main__":
    main()