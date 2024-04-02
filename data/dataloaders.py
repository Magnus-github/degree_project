import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.str_to_class import str_to_class


def get_dataloaders(cfg):
    train_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="train")
    val_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="val")

    labels_train = [train_dataset.labels[train_dataset.ids[int(file.split("_")[1])]] for file in train_dataset.data]
    y_train = [cfg.dataset.mapping[label] for label in labels_train]
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    # num_samples = len(samples_weight)
    # num_samples = int(max(class_sample_count)*len(class_sample_count))
    num_samples = int(min(class_sample_count)*len(class_sample_count))
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.hparams.batch_size, sampler=sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=cfg.hparams.batch_size, shuffle=True)

    return {"train": train_dataloader, "val": val_dataloader}
