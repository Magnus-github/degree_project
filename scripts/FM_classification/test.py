import torch
import matplotlib.pyplot as plt

from optimizer import CosineDecayWithWarmUpScheduler, CyclicCosineDecayLR
import sys
sys.path.append(".")
from data.dataloaders import get_dataloaders

import omegaconf



def test_learning_rate_schedule():
    cfg = omegaconf.OmegaConf.load("config/train_TF.yaml")

    optim = torch.optim.Adam([torch.tensor(1.0)], lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
    scheduler = CyclicCosineDecayLR(optim, **cfg.hparams.scheduler.params)

    lr = []
    for i in range(cfg.hparams.epochs):
        for j in range(cfg.hparams.scheduler.params_CSWU.step_per_epoch):
            optim.step()
        scheduler.step()
        lr.append(scheduler.get_last_lr()[0])
            

    print("Plotting...")
    print(len(lr))
    plt.figure()
    plt.plot(lr)
    plt.savefig("cosine_decay.png")



def plot_validation_samples():
    cfg = omegaconf.OmegaConf.load("config/train_AE.yaml")
    val_dataloader = get_dataloaders(cfg)['val']

    for b, data in enumerate(val_dataloader):
        inputs, pose_sequence = data[0][0], data[1][0]
        inputs = inputs.reshape(-1, 4, 18, 5)
        inputs = inputs.permute(0, 3, 1, 2)

        # pose_sequence = pose_sequence.reshape(-1, 2, 18)
        for x in range(10):
            compare = torch.abs(pose_sequence[x:x+5] - inputs[x])
            print(torch.max(compare))

        for n in range(0,2500,500):
            fig, axs = plt.subplots(18, 4, figsize=(15, 5*18))
            outdir = "output/debugging/features/"
            for i in range(4):
                for j in range(18):
                    axs[j, i].plot(inputs[n,:,i, j])
                    axs[j, i].set_title(f'Joint {j} - {["x", "y", "v_x", "v_y"][i]}')

            plt.savefig(f"{outdir}reconstruction_{b}_{n}.png")




if __name__ == "__main__":
    test_learning_rate_schedule()