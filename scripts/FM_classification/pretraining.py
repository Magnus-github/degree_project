import torch
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import logging

import sys
import os
sys.path.append(".")
from data.dataloaders import get_dataloaders, get_sparse_edge_matrix_skeleton_14
from scripts.utils.str_to_class import str_to_class
from scripts.utils.check_debugger import debugger_is_active

class Trainer:
    def __init__(self, cfg, dataloaders, run_id=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cfg = cfg
        self.output_dir = os.path.join(self._cfg.save_path, str(run_id))
        self.run_id = run_id
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = str_to_class(cfg.model.name)(**cfg.model.params).to(self.device)
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.optimizer = str_to_class(cfg.hparams.optimizer.name)(self.model.parameters(), **cfg.hparams.optimizer.params)
        self.scheduler = str_to_class(cfg.hparams.scheduler.name)(self.optimizer, **cfg.hparams.scheduler.params)
        # self.criterion = str_to_class(cfg.hparams.criterion.name)(**cfg.hparams.criterion.params)
        self.criterion = self.model.loss_function

        self.edge_indices = get_sparse_edge_matrix_skeleton_14().indices().to(self.device)

        self.seq_orig_dim = cfg.hparams.orig_dim

        if not debugger_is_active() and cfg.wandb.enable:
            wandb.watch(self.model)
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self):
        self.model.train()
        for epoch in range(self._cfg.hparams.epochs):
            running_loss = 0.0
            running_reconst_loss = 0.0
            running_kld = 0.0
            last_val_losses = []
            for i, data in enumerate(tqdm(self.train_loader)):
                inputs = data[0]#[:10]
                # N, C, J, t = inputs.shape
                # inputs = inputs.reshape(N, J, C*t)
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()
                if "GCN" in self._cfg.model.name:
                    outputs = self.model(inputs, self.edge_indices)
                else:
                    outputs = self.model(inputs)
                losses = self.criterion(outputs['pred'], inputs, outputs['distribution'])
                total_loss = losses['loss']
                reconstruction_loss = losses['Reconstruction_Loss']
                kld = losses['KLD']
                running_loss += total_loss.item()
                running_reconst_loss += reconstruction_loss.item()
                running_kld += kld.item()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if not debugger_is_active() and self._cfg.wandb.enable:
                    last_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({"Train Total Loss [batch]": total_loss.item(), "Train Reconstruction Loss [batch]": reconstruction_loss.item(),
                               "Train KL-Divergence [batch]":kld.item() , "Learning Rate": last_lr})
                
            self.logger.info(f"Epoch {epoch} Loss: {running_loss / len(self.train_loader)}")

            if self._cfg.plot_reconstruction.enable and epoch % self._cfg.plot_reconstruction.period == 0:
                if running_loss / len(self.train_loader) < 1.5:
                    self.plot_reconstruction(inputs, outputs['pred'], data_type="train")
                    self.plot_skeleton(inputs, outputs['pred'], data_type="train")

            val_loss, val_reconstruction_loss, val_kld = self.validate(epoch)
            
            if not debugger_is_active() and self._cfg.wandb.enable:
                wandb.log({"Train Loss [epoch]": running_loss / len(self.train_loader),
                           "Train Reconstruction Loss [epoch]": running_reconst_loss / len(self.train_loader),
                           "Train KL-Divergence [epoch]": running_kld / len(self.train_loader),
                           "Val Loss": val_loss, "Val Reconstruction Loss": val_reconstruction_loss,
                           "Val KL-Divergence": val_kld, "Epoch": epoch})

            last_val_losses.append(val_loss)
            if len(last_val_losses) > 5:
                if last_val_losses[-1] > last_val_losses[-2] > last_val_losses[-3] > last_val_losses[-4] > last_val_losses[-5]:
                    self.logger.info("Early stopping")
                    break

        torch.save(self.model.state_dict(), self.output_dir+"/model.pth")

    def validate(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_reconst_loss = 0.0
            running_val_kld = 0.0
            for i, data in enumerate(tqdm(self.val_loader)):
                inputs = data[0]
                # N, C, J, t = inputs.shape
                # inputs = inputs.reshape(N, J, C*t)
                inputs = inputs.to(self.device)
                if "GCN" in self._cfg.model.name:
                    outputs = self.model(inputs, self.edge_indices)
                else:
                    outputs = self.model(inputs)
                losses = self.criterion(outputs['pred'], inputs, outputs['distribution'])
                running_val_loss += losses['loss'].item()
                running_val_reconst_loss += losses['Reconstruction_Loss'].item()
                running_val_kld += losses['KLD'].item()

                if i == 0 and self._cfg.plot_reconstruction.enable and epoch % self._cfg.plot_reconstruction.period == 0:
                    if running_val_loss / len(self.val_loader) < 0.01:
                        self.plot_reconstruction(inputs, outputs['pred'], data_type="val")

        total_val_loss = running_val_loss / len(self.val_loader)
        val_reconstruction_loss = running_val_reconst_loss / len(self.val_loader)
        val_kld = running_val_kld / len(self.val_loader)
        self.logger.info(f"Validation Loss: {total_val_loss}")

        return total_val_loss, val_reconstruction_loss, val_kld
    
    def plot_reconstruction(self, inputs, pred, data_type="val"):
        input = inputs.cpu()
        input = input.reshape(-1, *self.seq_orig_dim)
        input = input.permute(0, 3, 1, 2).numpy()
        pred = pred.cpu()
        pred = pred.reshape(-1, *self.seq_orig_dim)
        pred = pred.permute(0, 3, 1, 2).detach().numpy()
        chans, num_joints, clip_len = self.seq_orig_dim
        seq_len = input.shape[0]
        for i in np.linspace(0, seq_len-1, 5).astype(int):
            fig, ax = plt.subplots(num_joints, chans, figsize=(15, 5*num_joints))
            for j in range(chans):
                for k in range(num_joints):
                    ax[k, j].plot(input[i,:,j,k], label="GT")
                    ax[k, j].plot(pred[i,:,j,k], label="Pred")
                    ax[k, j].set_title(f"Joint {k} - {['x', 'y', 'v_x', 'v_y'][j]}")
                    ax[k, j].legend()
            id = self.run_id if self.run_id is not None else "noID"
            plt.savefig(f"reconstruction_{id}_{data_type}_{i}.png")
            plt.close()

    def plot_skeleton(self, inputs, pred, data_type="val"):
        input = inputs.cpu()
        input = input.reshape(-1, *self.seq_orig_dim)
        input = input.permute(0, 3, 1, 2).numpy()
        pred = pred.cpu()
        pred = pred.reshape(-1, *self.seq_orig_dim)
        pred = pred.permute(0, 3, 1, 2).detach().numpy()
        chans, num_joints, clip_len = self.seq_orig_dim
        seq_len = input.shape[0]
        fig, ax = plt.subplots(5, 1, figsize=(5, 5*5))

        for j, i in enumerate(np.linspace(0, seq_len-1, 5).astype(int)):
            # for j in range(num_joints):
            ax[j].scatter(input[i,0,0,:], input[i,0,1,:], label="GT")
            ax[j].scatter(pred[i,0,0,:], pred[i,0,1,:], label="Pred")
            for edge in [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [2, 8], [8, 9], [9, 10], [8,11], [5, 11], [11, 12], [12, 13]]:
                ax[j].plot(input[i,0,0,edge], input[i,0,1,edge], color='black')
                ax[j].plot(pred[i,0,0,edge], pred[i,0,1,edge], color='gray')
            ax[j].set_xlim(-1, 1)
            ax[j].set_ylim(-1, 1)
            ax[j].set_title(f"Skeleton at frame {i} - xy")
            ax[j].legend()
        
        id = self.run_id if self.run_id is not None else "noID"
        plt.savefig(f"skeleton_{id}_{data_type}.png")
        plt.close()
            



def main():
    # Load the configuration file
    cfg = OmegaConf.load("config/train_AE.yaml")
    dataloaders = get_dataloaders(cfg)
    trainer = Trainer(cfg, dataloaders)
    trainer.train()


if __name__ == "__main__":
    main()