import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import omegaconf

from optimizer import CosineDecayWithWarmUpScheduler
import sys
sys.path.append(".")
from data.datasetKI import KIDataset
from scripts.utils.str_to_class import str_to_class


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def plot_lr():
    cfg = omegaconf.OmegaConf.load("config/train_TF.yaml")

    # optim = torch.optim.Adam([torch.tensor(1.0)], lr=1e-3)
    optimizer = str_to_class(cfg.hparams.optimizer.name)([torch.tensor(1.0)], **cfg.hparams.optimizer.params)
    scheduler = str_to_class(cfg.hparams.scheduler.name)(optimizer, **cfg.hparams.scheduler.params)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
    # scheduler = CosineDecayWithWarmUpScheduler(optim, **cfg.hparams.scheduler.params)

    for i in range(400):
        for j in range(6):
            optimizer.step()
            scheduler.step()
            

    print("Plotting...")
    print(len(scheduler.lr_list))
    plt.figure(figsize=(15, 10))
    plt.plot(scheduler.lr_list)
    plt.xlabel("Step", fontsize=45)
    plt.ylabel("Learning rate", fontsize=45)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.savefig("cosine_decay.pdf")



def plot_dataset_statistics(cfg: omegaconf.DictConfig):
    dataset = KIDataset(**cfg.dataset.params, mode="all")

    labels = dataset.labels
    print(labels)
    labelcount = {l: labels.count(l) for l in sorted(set(labels))}
    mapping = {1: 'FM-', 4: 'FM_abnormal', 12: 'FM+'}
    bar_labels = [mapping[l] for l in labelcount.keys()]
    print(labelcount)

    fig, ax = plt.subplots()
    ax.bar(bar_labels, labelcount.values())
    ax.set_ylabel('Number of samples')
    ax.set_title('Number of samples per class')
    plt.savefig("dataset_statistics.pdf")


def compare_classes(cfg: omegaconf.DictConfig):
    dataset = KIDataset(**cfg.dataset.params, mode="all")

    labels = dataset.labels
    labelcount = {l: labels.count(l) for l in sorted(set(labels))}

    FM_plus = []
    FM_minus = []
    FM_abnormal = []

    iterator = iter(dataset)

    for _ in range(len(dataset)):
        pose_sequence, label = next(iterator)
        if label == 1:
            FM_minus.append(pose_sequence)
        elif label == 4:
            FM_abnormal.append(pose_sequence)
        elif label == 12:
            FM_plus.append(pose_sequence)

    print(len(FM_plus))
    print(len(FM_minus))
    print(len(FM_abnormal))

    FM_plus = np.concatenate(FM_plus, axis=0)
    FM_minus = np.concatenate(FM_minus, axis=0)
    FM_abnormal = np.concatenate(FM_abnormal, axis=0)
    
    for j in range(FM_plus.shape[1]):
        ax, fig = plt.subplots()
        plt.boxplot(FM_plus[:, j, 0])
        plt.boxplot(FM_minus[:, j, 0])
        plt.boxplot(FM_abnormal[:, j, 0])

   
   
   
   
   
    # FM_plus_means = np.stack([FM_plus[i].mean(axis=0) for i in range(len(FM_plus))], axis=0)
    # FM_plus_means = FM_plus_means.mean(axis=0)
    # FM_minus_means = np.stack([FM_minus[i].mean(axis=0) for i in range(len(FM_minus))], axis=0)
    # FM_minus_means = FM_minus_means.mean(axis=0)
    # FM_abnormal_means = np.stack([FM_abnormal[i].mean(axis=0) for i in range(len(FM_abnormal))], axis=0)
    # FM_abnormal_means = FM_abnormal_means.mean(axis=0)

    
    


if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load("config/train_TF.yaml")
    # compare_classes(cfg)
    plot_lr()