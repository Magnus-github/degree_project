import torch
import numpy as np

from omegaconf import OmegaConf
import os
import logging
import wandb
from tqdm import tqdm

import pytz
import datetime
import sys
sys.path.append(os.curdir)

from scripts.utils.str_to_class import str_to_class
from data.dataloaders import get_dataloaders, get_dataloaders_clips
from scripts.utils.check_debugger import debugger_is_active


class Trainer:
    def __init__(self, cfg, dataloaders, run_id=None):
        self._cfg = cfg
        if run_id is None:
            run_id = "debug"
        else:
            datetime_str = datetime.datetime.now(pytz.timezone('Europe/Stockholm')).strftime("%Y-%m-%d_%H:%M:%S")
            run_id = datetime_str + "_" + run_id
        self.output_dir = os.path.join(self._cfg.outputs.path, str(run_id))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cfg.model.in_features == "kinematics_pos":
            self._cfg.model.in_params.joint_in_channels = 2
        elif cfg.model.in_features == "kinematics_vel":
            self._cfg.model.in_params.joint_in_channels = 4
        elif cfg.model.in_features == "kinematics_acc":
            self._cfg.model.in_params.joint_in_channels = 6
        self.model = str_to_class(cfg.model.name)(**cfg.model.in_params)
        if cfg.model.load_weights.enable:
            if cfg.test.enable:
               weights_path = cfg.test.model_weights
            else:
                weights_path = cfg.model.load_weights.path 
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['val']
        self.test_dataloader = dataloaders['test']
        self.criterion = str_to_class(cfg.hparams.criterion.name)(**cfg.hparams.criterion.params, device=self.device)
        self.optimizer = str_to_class(cfg.hparams.optimizer.name)(self.model.parameters(), **cfg.hparams.optimizer.params)
        if "Adam" in cfg.hparams.optimizer.name and cfg.hparams.optimizer.use_scheduler is False:
            self.scheduler = None
        else:
            self.scheduler = str_to_class(cfg.hparams.scheduler.name)(self.optimizer, **cfg.hparams.scheduler.params)
        self.model.to(self.device)
        if not debugger_is_active() and cfg.logger.enable:
            wandb.watch(self.model)
        self.class_mapping = cfg.dataset.mapping
        self.logger = logging.getLogger(self.__class__.__name__)

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
        elif "kinematics" in name:
            # Compute differences for both x and y dimensions
            step = self._cfg.dataset.diff_step
            t = step / self._cfg.dataset.fps
            diff = pose_sequence[:, step:] - pose_sequence[:, :-step]
            velocities = diff * t
            velocities = torch.concat([torch.zeros(velocities.shape[0], step, velocities.shape[2], velocities.shape[3]), velocities], dim=1)
            # scaled_velocities = velocities / torch.linalg.norm(velocities, axis=-1).unsqueeze(-1)

            # Compute diff of velocities in x and y
            diff_v = velocities[:, step:] - velocities[:, :-step]
            accelerations = diff_v / t
            accelerations = torch.concat([torch.zeros(accelerations.shape[0], step, accelerations.shape[2], accelerations.shape[3]), accelerations], dim=1)

            # calculate the total distance traveled by each joint in the sequence
            # distances = torch.zeros(velocities.shape)
            # for i in range(1, pose_sequence.shape[1]):
            #     distances[:, i, :, :] = distances[:, i-1, :, :] + velocities[:, i-1, :, :]*t + 0.5*accelerations[:, i-1, :, :]*t**2

            # distances = np.sqrt(distances[:, :, :, 0]**2 + distances[:, :, :, 1]**2)
            # distances = torch.linalg.norm(distances, axis=-1).unsqueeze(-1)

            # shape: [B, T, J, 6]
            features = torch.concat([pose_sequence, velocities, accelerations], dim=3)
            features = features.permute(0,1,3,2)
            # shape: [B, T, 6, J]
            features = features.float()

            joint_ids = []
            if "hands" in self._cfg.dataset.joints:
                joint_ids.extend([4, 7])
            if "feet" in self._cfg.dataset.joints:
                joint_ids.extend([10, 13])
            if "hips" in self._cfg.dataset.joints:
                joint_ids.extend([8, 11])
            if self._cfg.dataset.joints == "all":
                joint_ids = list(range(14))
            
            features = features[:, :, :, joint_ids]

            if name == "kinematics":
                return features
            elif name == "kinematics_pos":
                return features[:,:,:2,:]
            elif name == "kinematics_vel":
                return features[:,:,:4,:]
            elif name == "kinematics_acc":
                return features[:,:,:6,:]

        elif name == "basic":
            return pose_sequence[:,:,:,:2]
        else:
            self.logger.error(f"Feature type {name} not found... Using basic features.")
            return pose_sequence
        
    def one_hot_encode(self, target):
        one_hot = torch.zeros(len(target), 3)
        for i, t in enumerate(target):
            one_hot[i, self.mapping[t]] = 1
        return one_hot
        
    def train(self):
        self.model.train()
        self.logger.info("Starting training...")
        # num_samples = len(self.train_dataloader)*self._cfg.hparams.batch_size - 
        outputs = torch.zeros(len(self.train_dataloader)*self._cfg.hparams.batch_size, self._cfg.model.in_params.num_classes)
        # labels = torch.zeros(len(self.train_dataloader)*self._cfg.hparams.batch_size)
        labels = []
        val_losses = []
        x = np.arange(0, self._cfg.hparams.early_stopping.patience)
        val_loss = np.inf
        best_val_loss = np.inf
        # last_lr = self.scheduler.get_last_lr()[0]
        last_lr = self.optimizer.param_groups[0]['lr']
        for epoch in range(self._cfg.hparams.epochs):
            running_loss = 0.0
            for i, (data) in enumerate(tqdm(self.train_dataloader)):
                if len(data) == 3:
                    pose_sequence, target, count = data
                else:
                    pose_sequence, target = data
                if len(pose_sequence.shape) == 5:
                    B, n_samples, T, J, C = pose_sequence.shape
                    pose_sequence = pose_sequence.view(B*n_samples, T, J, C)

                label = [self.class_mapping[t.item()] for t in target]
                labels.extend(label)
                label = torch.tensor(label)
                # labels[i*self._cfg.hparams.batch_size:i*self._cfg.hparams.batch_size+len(label)] = label
                # pose_sequence = pose_sequence.to(self.device)
                label = label.to(self.device)

                assert pose_sequence.shape[0] == label.shape[0]

                self.optimizer.zero_grad()

                pose_sequence = pose_sequence.permute(0,3,1,2)
                features = self.create_features(pose_sequence, self._cfg.model.in_features)
                features = features.to(self.device)
                output = self.model(features)
                outputs[i*self._cfg.hparams.batch_size:i*self._cfg.hparams.batch_size+len(output)] = output.softmax(axis=1)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                running_loss += loss.item()

                # logger.info(f"Epoch {epoch}, batch {i}, loss: {loss.item()}")
                if not debugger_is_active() and self._cfg.logger.enable:
                    wandb.log({"Train Loss [batch]": loss.item()})

                if not debugger_is_active() and self._cfg.logger.enable:
                    if last_lr != self.optimizer.param_groups[0]['lr'] or i == 0:
                        last_lr = self.optimizer.param_groups[0]['lr']
                        wandb.log({"Learning Rate": last_lr})

            train_accuracy = self.compute_accuracy(outputs, labels)
            
            self.logger.info(f"Epoch {epoch} loss: {running_loss/len(self.train_dataloader)}")
            if not debugger_is_active() and self._cfg.logger.enable:
                wandb.log({"Epoch": epoch,
                    "Train Loss [epoch]": running_loss/len(self.train_dataloader),
                       "Train Accuracy": train_accuracy})

            if (epoch+1) % self._cfg.hparams.validation_period == 0 or epoch == 0:
                val_loss = self.validate(epoch)

                val_losses.append(val_loss)
                if len(val_losses) > self._cfg.hparams.early_stopping.patience:
                    val_losses.pop(0)

                    if self._cfg.hparams.early_stopping.enable and epoch > self._cfg.hparams.early_stopping.after_epoch:
                        y = np.array(val_losses)
                        slope, _ = np.polyfit(x, y, 1)
                        if slope > self._cfg.hparams.early_stopping.slope_threshold:
                            self.logger.info(f"Early stopping at epoch {epoch}.")
                            self.save_model(epoch=epoch)
                            break

                    if val_loss < best_val_loss and val_loss < self._cfg.hparams.save_best_threshold:
                        best_val_loss = val_loss
                        self.save_model(best=True)

            # if self.scheduler:
            #     if "ReduceLROnPlateau" in self._cfg.hparams.scheduler.name:
            #         self.scheduler.step(running_loss/len(self.train_dataloader))
            #     else:
            #         self.scheduler.step()

            if self._cfg.logger.enable and (epoch+1) % self._cfg.model.save_period == 0:
                if epoch == self._cfg.hparams.epochs - 1:
                    self.save_model()
                else:
                    self.save_model(epoch=epoch)
            
            running_loss = 0.0

        self.save_model()

        self.logger.info("Finished training, closing...")
        return

    def validate(self, epoch):
        self.model.eval()
        self.logger.info("Starting validation...")
        outputs = torch.zeros(len(self.val_dataloader), self._cfg.model.in_params.num_classes)
        labels = torch.zeros(len(self.val_dataloader))
        running_val_loss = 0.0
        with torch.no_grad():
            for i, (data) in enumerate(tqdm(self.val_dataloader)):
                if len(data) == 3:
                    pose_sequence, target, count = data
                else:
                    pose_sequence, target = data

                label = torch.tensor([self.class_mapping[t] for t in target])
                labels[i] = label
                # pose_sequence = pose_sequence.to(self.device)
                label = label.to(self.device)

                features = self.create_features(pose_sequence, self._cfg.model.in_features)
                features = features.to(self.device)
                output = self.model(features)
                outputs[i] = output.softmax(axis=1)
                val_loss = self.criterion(output, label)
                running_val_loss += val_loss.item()
                # _, predicted = torch.max(output.data, 1)
                # total += label.size(0)
                # correct += (predicted == label).sum().item()

        accuracy = self.compute_accuracy(outputs, labels)
        self.logger.info(f"Accuracy of the network on the {len(self.val_dataloader)} test sequences: {100*accuracy}%")
        self.logger.info(f"Validation loss: {running_val_loss/len(self.val_dataloader)}")
        # write output to file
        if epoch == 0:
            f = open(os.path.join(self.output_dir, "validation_outputs.txt"), "w")
        else:
            f = open(os.path.join(self.output_dir, "validation_outputs.txt"), "a")
        f.write('Validation outputs and labels {}: \n{} \n'.format((epoch+1)//self._cfg.hparams.validation_period, torch.cat((outputs, labels.unsqueeze(1)), 1)))
        f.close()

        if not debugger_is_active() and self._cfg.logger.enable:
            wandb.log({"val_accuracy": accuracy,
                   "val_loss": running_val_loss/len(self.val_dataloader)})
        
        self.model.train()

        return running_val_loss/len(self.val_dataloader)
    
    def test(self):
        self.model.eval()
        self.logger.info("Starting testing...")
        outputs = torch.zeros(len(self.test_dataloader), self._cfg.model.in_params.num_classes)
        labels = torch.zeros(len(self.test_dataloader))
        ids = torch.zeros(len(self.test_dataloader))
        with torch.no_grad():
            for i, (data) in enumerate(self.test_dataloader):
                if len(data) == 3:
                    pose_sequence, target, id = data
                else:
                    pose_sequence, target = data

                label = torch.tensor([self.class_mapping[t] for t in target])
                labels[i] = label
                # pose_sequence = pose_sequence.to(self.device)
                label = label.to(self.device)

                ids[i] = id

                features = self.create_features(pose_sequence, self._cfg.model.in_features)
                features = features.to(self.device)
                print(id)
                # if i == 0:^
                output = self.model(features)
                print(output)
                outputs[i] = output.softmax(axis=1)

        accuracy = self.compute_accuracy(outputs, labels)
        self.logger.info(f"Accuracy of the network on the {len(self.test_dataloader)} test sequences: {100*accuracy}%")
        # write output to file
        f = open(os.path.join(self.output_dir, "test_outputs.txt"), "w")
        f.write('Test outputs and labels: \n{} \n'.format(torch.cat((outputs, labels.unsqueeze(1), ids.unsqueeze(1)), 1)))
        f.write(f"Accuracy of the network on the {len(self.test_dataloader)} test sequences: {100*accuracy}%")
        f.close()
        return

    def compute_accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct / total

    def save_model(self, epoch=None, best=False):
        path = os.path.join(self.output_dir, "model")
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = self._cfg.model.name.split(":")[-1]
        if epoch is None:
            id = "final"
        else:
            id = epoch
        if best:
            id = "best"
        torch.save(self.model.state_dict(), os.path.join(path, f"{model_name}_{id}.pth"))


if __name__ == "__main__":
    cfg = OmegaConf.load("config/train.yaml")
    if cfg.test.enable:
        cfg.logger.enable = False

    dataloaders = get_dataloaders(cfg)

    if debugger_is_active():
        cfg.hparams.epochs = 1
        cfg.hparams.validation_period = 1
        cfg.logger.enable = False
    elif cfg.logger.enable:
        run = wandb.init(project='FM-classification', config=dict(cfg))
    else:
        cfg.hparams.epochs = 25
        cfg.hparams.validation_period = 1

    trainer = Trainer(cfg, dataloaders, run_id=run.name if cfg.logger.enable else None)

    if cfg.test.enable:
        cfg.model.load_weights.enable = True
        trainer.test()
    else:
        trainer.train()

    if cfg.logger.enable:
        wandb.finish()
