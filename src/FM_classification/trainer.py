import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from omegaconf import OmegaConf
import os
import json
import logging
import coloredlogs
import wandb
from tqdm import tqdm

import sys
sys.path.append(os.curdir)

from src.utils.str_to_class import str_to_class

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


class Trainer:
    def __init__(self, cfg, dataloaders):
        self._cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = str_to_class(cfg.model.name)(**cfg.model.in_params)
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['val']
        self.criterion = str_to_class(cfg.hparams.criterion.name)(**cfg.hparams.criterion.params, device=self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), **cfg.hparams.optimizer.params)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, **cfg.hparams.scheduler.params)
        self.model.to(self.device)
        if not debugger_is_active() and cfg.logger.enable:
            wandb.watch(self.model)
        self.class_mapping = cfg.dataset.mapping

    def create_features(self, pose_sequence, name):
        if name == "distance_mat":
            # Reshape pose_sequence to (B, T, J, 1, C)
            pose_sequence_reshaped = pose_sequence.unsqueeze(3)
            pose_sequence_reshaped = pose_sequence_reshaped[:, :, :14, :, :2]

            # Compute absolute differences for both x and y dimensions
            abs_diff = torch.abs(pose_sequence_reshaped - pose_sequence_reshaped.permute(0, 1, 3, 2, 4))

            # shape: [B, T, C, J, J]
            features = abs_diff.permute(0, 1, 4, 2, 3)

            features = features.float()

            return features
        elif name == "kinematics":
            # Compute differences for both x and y dimensions
            diff = pose_sequence[:, 1:, :, :-1] - pose_sequence[:, :-1, :, :-1]
            velocities = diff / (1/self._cfg.data.fps)
            velocities = torch.concat([torch.zeros(velocities.shape[0], 1, velocities.shape[2], velocities.shape[3]), velocities], dim=1)

            # Compute diff of velocities in x and y
            diff_v = velocities[:, 1:] - velocities[:, :-1]
            accelerations = diff_v / (1/self._cfg.data.fps)
            accelerations = torch.concat([torch.zeros(accelerations.shape[0], 1, accelerations.shape[2], accelerations.shape[3]), accelerations], dim=1)

            # TODO: calculate depth based on amount of pixes between joints
            depth = torch.zeros_like(pose_sequence[:,:,:,1])

            # shape: [B, T, J, 6]
            features = torch.concat([pose_sequence[:,:,:,:2], velocities, accelerations], dim=3)

            return features
        elif name == "basic":
            return pose_sequence[:,:,:,:2]
        else:
            logger.error(f"Feature type {name} not found... Using basic features.")
            return pose_sequence
        
    def one_hot_encode(self, target):
        mapping = {'1': 0, '4': 1, '12': 2}
        one_hot = torch.zeros(len(target), 3)
        for i, t in enumerate(target):
            one_hot[i, mapping[t]] = 1
        return one_hot
        
    def train(self):
        self.model.train()
        logger.info("Starting training...")
        for epoch in range(cfg.hparams.epochs):
            running_loss = 0.0
            for i, (data, target) in enumerate(tqdm(self.train_dataloader)):
                pose_sequence = data
                # label = self.one_hot_encode(target)
                label = torch.tensor([self.class_mapping[t] for t in target])
                pose_sequence = pose_sequence.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                features = self.create_features(pose_sequence, self._cfg.model.in_features)
                output = self.model(features)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # logger.info(f"Epoch {epoch}, batch {i}, loss: {loss.item()}")
                if not debugger_is_active() and self._cfg.logger.enable:
                    wandb.log({"Train Loss [batch]": loss.item()})
            
            logger.info(f"Epoch {epoch} loss: {running_loss/len(self.train_dataloader)}")
            if not debugger_is_active() and self._cfg.logger.enable:
                wandb.log({"Epoch": epoch,
                       "Train Loss [epoch]": running_loss/len(self.train_dataloader)})
            
            running_loss = 0.0

            self.scheduler.step()

            if (epoch+1) % self._cfg.hparams.validation_period == 0 or epoch == 0:
                self.validate(epoch)

        self.save_model()

    def validate(self, epoch):
        self.model.eval()
        logger.info("Starting validation...")
        correct = 0
        total = 0
        outputs = torch.zeros(len(self.val_dataloader), 3)
        labels = torch.zeros(len(self.val_dataloader))
        running_val_loss = 0.0
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.val_dataloader)):
                pose_sequence = data
                label = torch.tensor([self.class_mapping[t] for t in target])
                labels[i] = label
                pose_sequence = pose_sequence.to(self.device)
                label = label.to(self.device)

                features = self.create_features(pose_sequence, self._cfg.model.in_features)
                output = self.model(features)
                outputs[i] = output.softmax(axis=1)
                val_loss = self.criterion(output, label)
                running_val_loss += val_loss.item()
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        logger.info(f"Accuracy of the network on the {total} test sequences: {100 * correct / total}%")
        logger.info(f"Validation loss: {running_val_loss/len(self.val_dataloader)}")
        # write output to file
        if epoch == 0:
            f = open("output/FM/validation_outputs.txt", "w")
        else:
            f = open("output/FM/validation_outputs.txt", "a")
        f.write('Validation outputs and labels {}: \n{} \n'.format((epoch+1)//self._cfg.hparams.validation_period, torch.cat((outputs, labels.unsqueeze(1)), 1)))
        f.close()

        if not debugger_is_active() and self._cfg.logger.enable:
            wandb.log({"val_accuracy": 100 * correct / total,
                   "val_loss": running_val_loss/len(self.val_dataloader),
                   "val_outputs": outputs})
        
        self.model.train()

    def save_model(self):
        torch.save(self.model.state_dict(), self._cfg.hparams.model_path)


if __name__ == "__main__":
    cfg = OmegaConf.load("config/train.yaml")

    train_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="train")
    val_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="val")

    labels_train = [train_dataset.labels[train_dataset.ids[int(file.split("_")[1])]] for file in train_dataset.data]
    y_train = [cfg.dataset.mapping[label] for label in labels_train]
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    # num_samples = len(samples_weight)
    num_samples = int(max(class_sample_count)*len(class_sample_count))
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.hparams.batch_size, sampler=sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=cfg.hparams.batch_size, shuffle=True)

    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    if debugger_is_active():
        cfg.hparams.epochs = 1
        cfg.hparams.validation_period = 1
    elif cfg.logger.enable:
        wandb.init(project='FM-classification', config=dict(cfg))
    else:
        cfg.hparams.epochs = 10
        cfg.hparams.validation_period = 1

    trainer = Trainer(cfg, dataloaders)

    trainer.train()

    wandb.finish()
