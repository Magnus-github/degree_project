import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import argparse
import logging
import coloredlogs

from data.dataloaders import get_dataloaders, get_dataloaders_clips
from scripts.FM_classification.trainer import Trainer
from scripts.utils.check_debugger import debugger_is_active

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] [%(name)s] [%(module)s] [%(levelname)s] %(message)s")

def main(cfg: DictConfig):
    if cfg.test.enable:
        cfg.logger.enable = False

    if "dynamicClipSample" in cfg.dataset.name:
        dataloaders = get_dataloaders_clips(cfg)
    else:
        dataloaders = get_dataloaders(cfg)

    if debugger_is_active():
        cfg.hparams.epochs = 1
        cfg.hparams.validation_period = 1
    elif cfg.logger.enable:
        run = wandb.init(project='FM-classification_2Class', config=dict(cfg))
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

    logger.info("Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    logger.info(cfg)
    main(cfg)
    
    # magnitude = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # probs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # for mag in magnitude:
    #     for p in probs:
    #         if mag == 0.1 and p == 0.5 or mag == 0.1 and p == 0.6:
    #             continue
    #         cfg.dataset.transform.params.magnitude = mag
    #         cfg.dataset.transform.params.p = p
    #         main(cfg)
