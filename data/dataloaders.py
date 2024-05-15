import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.utils.str_to_class import str_to_class


def get_dataloaders(cfg):
    if cfg.dataset.transform.enable:
        transform = str_to_class(cfg.dataset.transform.name)(**cfg.dataset.transform.params)
    else:
        transform = None
    train_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="train", transform=transform)
    val_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="val")
    test_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="test")

    labels_train = [train_dataset.labels[train_dataset.ids[int(file.split("_")[1])]] for file in train_dataset.data]
    y_train = [cfg.dataset.mapping[label] for label in labels_train]
    if cfg.dataset.sampling.enable:
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        if cfg.dataset.sampling.method == "oversampling":
            num_samples = int(max(class_sample_count)*len(class_sample_count))
        elif cfg.dataset.sampling.method == "undersampling":
            num_samples = int(min(class_sample_count)*len(class_sample_count))
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples)
    else:
        sampler = None
        label_distribution = {label: y_train.count(label) for label in set(y_train)}
        print(f"Label distribution: {label_distribution}")


    train_dataloader = DataLoader(train_dataset, batch_size=cfg.hparams.batch_size, sampler=sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


def get_dataloaders_clips(cfg):
    if cfg.dataset.transform.enable:
        transform = str_to_class(cfg.dataset.transform.name)(**cfg.dataset.transform.params)
    else:
        transform = None
    train_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="train", transform=transform, **cfg.dataset.params_clips)
    dataset_name = cfg.dataset.name.split("_")[0]
    val_dataset = str_to_class(dataset_name)(**cfg.dataset.params, mode="val")
    test_dataset = str_to_class(dataset_name)(**cfg.dataset.params, mode="test")

    labels_train = [train_dataset.labels[train_dataset.ids[int(file.split("_")[1])]] for file in train_dataset.data]
    y_train = [cfg.dataset.mapping[label] for label in labels_train]
    if cfg.dataset.sampling.enable:
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        if cfg.dataset.sampling.method == "oversampling":
            num_samples = int(max(class_sample_count)*len(class_sample_count))
        elif cfg.dataset.sampling.method == "undersampling":
            num_samples = int(min(class_sample_count)*len(class_sample_count))
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples)
    else:
        sampler = None
        label_distribution = {label: y_train.count(label) for label in set(y_train)}
        print(f"Label distribution: {label_distribution}")


    train_dataloader = DataLoader(train_dataset, batch_size=cfg.hparams.batch_size, sampler=sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


def get_sparse_edge_indices_14():
    matrix = torch.tensor([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ]).float().to_sparse()

    return matrix.indices()


def collate_fn(batch):
    pose_sequences = [item[0] for item in batch]

    pose_sequences = torch.concat(pose_sequences, dim=0)
    labels = [item[1] for item in batch]
    labels = torch.concat(labels, dim=0)
    return pose_sequences, labels