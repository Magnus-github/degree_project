import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import argparse
import logging
import coloredlogs

from data.dataloaders import get_dataloaders, get_dataloaders_clips
from scripts.FM_classification.trainer import Trainer
from scripts.FM_classification.pretraining import Trainer as Pretrainer
from scripts.utils.check_debugger import debugger_is_active

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")

def train(cfg: DictConfig):
    if cfg.test.enable:
        cfg.logger.enable = False

    if "dynamicClipSample" in cfg.dataset.name:
        dataloaders = get_dataloaders_clips(cfg, fold=0)
    else:
        dataloaders = get_dataloaders(cfg, fold=0)

    if debugger_is_active():
        cfg.hparams.epochs = 1
        cfg.hparams.validation_period = 1
    elif cfg.logger.enable:
        if "TimeFormer" in cfg.model.name:
            suffix = "TimeFormer"
        elif "STTransformer" in cfg.model.name:
            suffix = "STTransformer"
        else:
            raise ValueError("Unknown model name.")
        run = wandb.init(project=f'{cfg.logger.project}_{suffix}', config=dict(cfg), entity="m46nu5")
    else:
        cfg.hparams.epochs = 25
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
        trainer.train()
        logger.info("Training finished.")

    if cfg.logger.enable:
        wandb.finish()

    logger.info("Done!")


def pretrain(cfg: DictConfig):
    # if cfg.test.enable:
    #     cfg.logger.enable = False

    dataloaders = get_dataloaders(cfg)

    if debugger_is_active():
        cfg.hparams.epochs = 1
        cfg.hparams.validation_period = 1
        cfg.wandb.enable = False
    elif cfg.wandb.enable:
        if "VAE" in cfg.model.name:
            suffix = "VAE"
        else:
            raise ValueError("Unknown model name.")
        run = wandb.init(project=f'Pretraining_{suffix}', config=dict(cfg), entity="m46nu5")
    else:
        cfg.hparams.epochs = 25
        cfg.hparams.validation_period = 1

    run_id = run.name if cfg.wandb.enable else None
    # if cfg.test.enable:
    #     cfg.model.load_weights.enable = True
    #     cfg.model.in_params.dropout = 0.0
    
    logger.info("Instantiating Trainer...")
    pretrainer = Pretrainer(cfg, dataloaders, run_id=run_id)

    # if cfg.test.enable:
    #     pretrainer.test()
    #     logger.info("Testing finished.")
    # else:
    #     pretrainer.train()
    #     logger.info("Pretraining finished."
    
    pretrainer.train()

    if cfg.wandb.enable:
        wandb.finish()

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_TF.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    logger.info(cfg)

    if "VAE" in cfg.model.name:
        pretrain(cfg)
    else:    
        train(cfg)
    
    # magnitude = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # probs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # for mag in magnitude:
    #     for p in probs:
    #         if mag == 0.1 and p == 0.5 or mag == 0.1 and p == 0.6:
    #             continue
    #         cfg.dataset.transform.params.magnitude = mag
    #         cfg.dataset.transform.params.p = p
    #         main(cfg)
