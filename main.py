import wandb
from omegaconf import OmegaConf

from data.dataloaders import get_dataloaders
from scripts.FM_classification.trainer import Trainer
from scripts.utils.check_debugger import debugger_is_active


def main(cfg):
    if cfg.test.enable:
        cfg.logger.enable = False

    dataloaders = get_dataloaders(cfg)

    if debugger_is_active():
        cfg.hparams.epochs = 1
        cfg.hparams.validation_period = 1
    elif cfg.logger.enable:
        run = wandb.init(project='FM-classification', config=dict(cfg))
    else:
        cfg.hparams.epochs = 25
        cfg.hparams.validation_period = 1

    run_id = run.name if cfg.logger.enable else None
    print(f"Run ID: {run_id}")
    if cfg.test.enable:
        cfg.model.load_weights.enable = True
        cfg.model.in_params.dropout = 0.0
    
    trainer = Trainer(cfg, dataloaders, run_id=run_id)

    if cfg.test.enable:
        trainer.test()
    else:
        trainer.train()

    if cfg.logger.enable:
        wandb.finish()


if __name__ == "__main__":
    cfg = OmegaConf.load("config/train.yaml")
    
    main(cfg)
