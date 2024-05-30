import numpy as np
import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig

import sys
sys.path.append(".")
from scripts.utils.str_to_class import str_to_class
# def str_to_class(string: str):
#     module_name, object_name = string.split(":")
#     module = __import__(module_name, fromlist=[object_name])
#     return getattr(module, object_name)


def get_dataloaders(cfg: DictConfig, test_fold: int = 0, val_fold: int = 0):
    if cfg.dataset.transform.enable:
        transform = str_to_class(cfg.dataset.transform.name)(**cfg.dataset.transform.params)
    else:
        transform = None
    train_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="train", transform=transform, test_fold=test_fold, val_fold=val_fold,)
    val_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="val", test_fold=test_fold, val_fold=val_fold)
    test_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="test", test_fold=test_fold, val_fold=val_fold)

    # labels_train = [train_dataset.labels[train_dataset.ids[int(file.split("_")[1])]] for file in train_dataset.data]
    labels_train = train_dataset.labels
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


def get_dataloaders_clips(cfg: DictConfig, test_fold: int = 0, val_fold: int = 0):
    if cfg.dataset.transform.enable:
        transform = str_to_class(cfg.dataset.transform.name)(**cfg.dataset.transform.params)
    else:
        transform = None
    train_dataset = str_to_class(cfg.dataset.name)(**cfg.dataset.params, mode="train", transform=transform, test_fold=test_fold, val_fold=val_fold, **cfg.dataset.params_clips)
    dataset_name = cfg.dataset.name.split("_")[0]
    val_dataset = str_to_class(dataset_name)(**cfg.dataset.params, mode="val", test_fold=test_fold, val_fold=val_fold)
    test_dataset = str_to_class(dataset_name)(**cfg.dataset.params, mode="test", test_fold=test_fold, val_fold=val_fold)

    for f in train_dataset.data:
        for fv in val_dataset.data:
            if f == fv:
                print("WHAAAT", f, fv)
        for ft in test_dataset.data:
            if f == ft:
                print("THIS IS A CATASTROPHE", f, ft)

    # labels_train = [train_dataset.labels[train_dataset.ids[int(file.split("_")[1])]] for file in train_dataset.data]
    labels_train = train_dataset.labels
    mapping = {1: 0, 4: 1, 12: 2}
    y_train = [mapping[label] for label in labels_train]
    if cfg.dataset.sampling.enable:
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        if cfg.dataset.sampling.method == "oversampling":
            num_samples = int(max(class_sample_count)*len(class_sample_count))
        elif cfg.dataset.sampling.method == "undersampling":
            num_samples = int(min(class_sample_count)*len(class_sample_count))
        elif cfg.dataset.sampling.method == "mixed":
            num_samples = int(len(y_train))
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), num_samples)
    else:
        sampler = None
        label_distribution = {label: y_train.count(label) for label in set(y_train)}
        print(f"Label distribution: {label_distribution}")


    train_dataloader = DataLoader(train_dataset, batch_size=cfg.hparams.batch_size, sampler=sampler, collate_fn=collate_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


def collate_fn(batch):
    pose_sequences = [item[0] for item in batch]

    pose_sequences = torch.concat(pose_sequences, dim=0)
    labels = [item[1] for item in batch]
    labels = torch.concat(labels, dim=0)
    return pose_sequences, labels


def get_edge_indices_14():
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



import pandas as pd
def get_label_distribution(cfg: DictConfig, test_fold: int = 0, val_fold: int = 0):
    cfg.hparams.batch_size = 1
    cfg.dataset.sampling.method = "mixed"
    # cfg.dataset.sampling.enable = False
    # cfg.dataset.mapping = {1: 0, 4: 1, 12: 2}
    dataloaders = get_dataloaders_clips(cfg, test_fold, val_fold)

    train_dataloader = dataloaders["train"]

    average_ranges = pd.DataFrame(columns=["1_x", "1_y", "4_x", "4_y", "12_x", "12_y"])
    x_1 = []
    y_1 = []
    x_4 = []
    y_4 = []
    x_12 = []
    y_12 = []
    label_count = {1: 0, 4: 0, 12: 0}
    for pose_sequence, labels in train_dataloader:
        for label in labels:
            label_count[label.item()] += 1
        
        # calculate range of amplitude
        min_A,_ = pose_sequence.min(dim=1)
        max_A,_ = pose_sequence.max(dim=1)
        range_A = max_A - min_A
        avg_range = range_A.mean(dim=0)

        if labels[0].item() == 1:
            if len(x_1) == 0:
                x_1 = avg_range[:,0].tolist()
                y_1 = avg_range[:,1].tolist()
            else:
                x_1 = [(x+x_new)/2 for x,x_new in zip(x_1,avg_range[:,0].tolist())]
                y_1 = [(y+y_new)/2 for y,y_new in zip(y_1,avg_range[:,1].tolist())]
        elif labels[0].item() == 4:
            if len(x_4) == 0:
                x_4 = avg_range[:,0].tolist()
                y_4 = avg_range[:,1].tolist()
            else:
                x_4 = [(x+x_new)/2 for x,x_new in zip(x_4,avg_range[:,0].tolist())]
                y_4 = [(y+y_new)/2 for y,y_new in zip(y_4,avg_range[:,1].tolist())]
        elif labels[0].item() == 12:
            if len(x_12) == 0:
                x_12 = avg_range[:,0].tolist()
                y_12 = avg_range[:,1].tolist()
            else:
                x_12 = [(x+x_new)/2 for x,x_new in zip(x_12,avg_range[:,0].tolist())]
                y_12 = [(y+y_new)/2 for y,y_new in zip(y_12,avg_range[:,1].tolist())]
    
    average_ranges["1_x"] = x_1
    average_ranges["1_y"] = y_1
    average_ranges["4_x"] = x_4
    average_ranges["4_y"] = y_4
    average_ranges["12_x"] = x_12
    average_ranges["12_y"] = y_12   

    print(label_count)
    relative_label_count = {label: count/sum(label_count.values()) for label, count in label_count.items()}
    print(relative_label_count)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/train_TF.yaml")
    get_label_distribution(cfg)