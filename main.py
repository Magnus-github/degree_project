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

def train_and_eval(cfg: DictConfig, fold: int = 0):
    if cfg.test.enable:
        cfg.logger.enable = False

    if "dynamicClipSample" in cfg.dataset.name:
        dataloaders = get_dataloaders_clips(cfg, fold=fold)
    else:
        dataloaders = get_dataloaders(cfg, fold=fold)

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
        run = wandb.init(project=f'FM-classification_2Class_{suffix}', config=dict(cfg), entity="m46nu5")
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
        metrics = trainer.train()
        logger.info("Training finished.")

    if cfg.logger.enable:
        wandb.finish()

    logger.info("Done!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_TF.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    logger.info(cfg)

    folds = range(10)
    all_metrics = {f"fold_{fold}": {} for fold in folds}
    for fold in range(cfg.dataset.params.num_folds):
        metrics_fold = train_and_eval(cfg, fold=fold)
        all_metrics[f"fold_{fold}"] = metrics_fold
    
    # magnitude = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # probs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # for mag in magnitude:
    #     for p in probs:
    #         if mag == 0.1 and p == 0.5 or mag == 0.1 and p == 0.6:
    #             continue
    #         cfg.dataset.transform.params.magnitude = mag
    #         cfg.dataset.transform.params.p = p
    #         main(cfg)
