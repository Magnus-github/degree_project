import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
import os
import logging
import coloredlogs
from tqdm import tqdm

import sys
sys.path.append(os.curdir)

from src.FM_classification.model import STTransformer
from src.utils.str_to_class import str_to_class

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")


class Trainer:
    def __init__(self, cfg, dataloader):
        self._cfg = cfg
        self.model = str_to_class(cfg.model.name)(**cfg.model.in_params)
        self.dataloader = dataloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), **cfg.hparams.optimizer.params)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, **cfg.hparams.scheduler.params)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def create_features(self, pose_sequence, name):
        if name == "distance_mat":
            # Reshape pose_sequence to (B, T, J, 1, C)
            pose_sequence_reshaped = pose_sequence.unsqueeze(3)

            # Compute absolute differences for both x and y dimensions
            abs_diff = torch.abs(pose_sequence_reshaped - pose_sequence_reshaped.permute(0, 1, 3, 2, 4))

            # shape: [B, T, C, J, J]
            features = abs_diff.permute(0, 1, 4, 2, 3)[:,:,:2,:,:]

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
            for i, (data, target) in enumerate(tqdm(self.dataloader)):
                pose_sequence = data
                label = self.one_hot_encode(target)
                pose_sequence = pose_sequence.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                features = self.create_features(pose_sequence, self._cfg.model.in_features)
                output = self.model(features)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                logger.info(f"Epoch {epoch}, batch {i}, loss: {loss.item()}")
            
            logger.info(f"Epoch {epoch} loss: {running_loss/len(self.dataloader)}")
            running_loss = 0.0



if __name__ == "__main__":
    cfg = OmegaConf.load("config/train.yaml")

    dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params)
    dataloader = DataLoader(dataset, batch_size=cfg.hparams.batch_size, shuffle=True)

    trainer = Trainer(cfg, dataloader)

    trainer.train()