import torch
import matplotlib.pyplot as plt

from optimizer import CosineDecayWithWarmUpScheduler

import omegaconf



def main():
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

if __name__ == "__main__":
    main()