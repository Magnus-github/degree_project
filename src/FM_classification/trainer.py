import torch
import torch.nn as nn

from omegaconf import OmegaConf
import os
import logging
import coloredlogs

from src.FM_classification.model import STTransformer
from src.utils.str_to_class import str_to_class

logger = logging.getLogger("Trainer")
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")


class Trainer:
    def __init__(self, cfg, dataloader):
        self._cfg = cfg
        self.model = str_to_class(cfg.model.name)(**cfg.model.in_params)
        self.dataloader = dataloader
        # self.optimizer = cfg.optimizer
        # self.criterion = cfg.criterion
        # self.device = cfg.device

    def create_features(self, pose_sequence, name):
        if name == "distance_mat":
            # Reshape pose_sequence to (B, T, J, 1, C)
            pose_sequence_reshaped = pose_sequence.unsqueeze(3)

            # Compute absolute differences for both x and y dimensions
            abs_diff = torch.abs(pose_sequence_reshaped - pose_sequence_reshaped.permute(0, 1, 3, 2, 4))

            # shape: [B, T, C, J, J]
            features = abs_diff.permute(0, 1, 4, 2, 3)[:,:,:2,:,:]

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
        
    def train(self):
        for epoch in range(cfg.hparams.epochs):
            running_loss = 0.0
            for i, (data, target) in enumerate(self.dataloader):
                pose_sequence = data
                label = target

                features = self.create_features(pose_sequence, self._cfg.model.in_features)
                output = self.model(features)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f"Epoch {epoch}, batch {i}, loss: {loss.item()}")



if __name__ == "__main__":
    cfg = OmegaConf.load("config/train.yaml")

    trainer = Trainer(cfg)

    print(trainer)