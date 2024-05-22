import wandb
from omegaconf import OmegaConf
from omegaconf import DictConfig
import argparse
import logging
import coloredlogs
import json
import os
import numpy as np

from data.dataloaders import get_dataloaders, get_dataloaders_clips
from scripts.FM_classification.trainer import Trainer
from scripts.utils.check_debugger import debugger_is_active

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")


def train_and_eval(cfg: DictConfig, test_fold: int = 0, val_fold: int = 0, project_name: str = "FM_classification"):
    if cfg.test.enable:
        cfg.logger.enable = False

    if "dynamicClipSample" in cfg.dataset.name:
        dataloaders = get_dataloaders_clips(cfg, test_fold=test_fold, val_fold=val_fold)
    else:
        dataloaders = get_dataloaders(cfg, test_fold=test_fold, val_fold=val_fold)

    if debugger_is_active():
        cfg.hparams.epochs = 1
        cfg.hparams.validation_period = 1
    elif cfg.logger.enable:
        if "TimeFormer" in cfg.model.name:
            suffix = "TimeFormer"
            if "GCN" in cfg.model.name:
                suffix += "_GCN"
        elif "STTransformer" in cfg.model.name:
            suffix = "STTransformer"
        elif "SMNN" in cfg.model.name:
            suffix = "SMNN"
        elif "TimeConvNet" in cfg.model.name:
            suffix = "TimeConvNet"    
        else:
            raise ValueError("Unknown model name.")
        run = wandb.init(project=f'{project_name}_{suffix}', config=dict(cfg), entity="m46nu5")
    else:
        cfg.hparams.epochs = 2
        cfg.hparams.validation_period = 1

    run_id = run.name if cfg.logger.enable else None
    if cfg.test.enable:
        cfg.model.load_weights.enable = True
        cfg.model.in_params.dropout = 0.0
    
    logger.info("Instantiating Trainer...")
    trainer = Trainer(cfg, dataloaders, run_id=run_id)

    if cfg.test.enable:
        trainer.test()
        logger.info("Testing finished.")
    else:
        metrics = trainer.train()
        logger.info("Training finished.")

    if cfg.logger.enable:
        wandb.finish()

    logger.info("Done!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_TF.yaml", help="Path to config file.")
    parser.add_argument("--project", type=str, default="FM_classification", help="Wandb project name.")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    logger.info(cfg)

    folds = range(cfg.dataset.params.num_folds)
    all_metrics = {}
    for test_fold in folds:
        all_metrics[f"test_fold_{test_fold}"] = {}
        for val_fold in folds[:-1]:
            metrics_fold = train_and_eval(cfg, test_fold=test_fold, val_fold=val_fold, project_name=args.project)
            all_metrics[f"test_fold_{test_fold}"][f"val_fold_{val_fold}"] = metrics_fold

            print(all_metrics)

    logger.info(all_metrics)

    # save metrics
    save_dir = "output/" + args.project
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "metrics.npy"), all_metrics, allow_pickle=True)

    # load with: np.load(os.path.join(save_dir, "metrics.npy"), allow_pickle=True).item()



    # magnitude = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # probs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # for mag in magnitude:
    #     for p in probs:
    #         if mag == 0.1 and p == 0.5 or mag == 0.1 and p == 0.6:
    #             continue
    #         cfg.dataset.transform.params.magnitude = mag
    #         cfg.dataset.transform.params.p = p
    #         main(cfg)
