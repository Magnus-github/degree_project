import torch
import matplotlib.pyplot as plt

import omegaconf

from optimizer import CosineDecayWithWarmUpScheduler
import sys
sys.path.append("/Midgard/home/tibbe/thesis/degree_project")
from data.datasetKI import KIDataset



def plot_lr():
    cfg = omegaconf.OmegaConf.load("/Midgard/home/tibbe/thesis/degree_project/config/train_TF.yaml")

    optim = torch.optim.Adam([torch.tensor(1.0)], lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
    scheduler = CosineDecayWithWarmUpScheduler(optim, **cfg.hparams.scheduler.params)

    for i in range(300):
        for j in range(120):
            optim.step()
            scheduler.step()
            

    print("Plotting...")
    print(len(scheduler.lr_list))
    plt.figure()
    plt.plot(scheduler.lr_list)
    plt.savefig("cosine_decay.png")



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
    plt.savefig("dataset_statistics.png")

    


if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load("/Midgard/home/tibbe/thesis/degree_project/config/train_TF.yaml")
    plot_dataset_statistics(cfg)