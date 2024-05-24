import os
from omegaconf import OmegaConf


def launch_experiment(name: str):
    os.system(f"sbatch --export PROJECT_NAME={name} train_TF")
    return


def main(cfg_path: str):
    # Load the configuration file
    default_cfg = OmegaConf.load(cfg_path)
    name = 'FM_cls_base'
    launch_experiment(name)

    # only hands
    cfg = default_cfg
    cfg.model.in_params.num_joints = 2
    cfg.dataset.joints = 'hands'
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_handsonly'
    launch_experiment(name)

    # only feet
    cfg = default_cfg
    cfg.model.in_params.num_joints = 2
    cfg.dataset.joints = 'feet'
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_feetonly'
    launch_experiment(name)

    # only hips
    cfg = default_cfg
    cfg.model.in_params.num_joints = 2
    cfg.dataset.joints = 'hips'
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_hipsonly'
    launch_experiment(name)

    # hands_feet_hips
    cfg = default_cfg
    cfg.model.in_params.num_joints = 6
    cfg.dataset.joints = 'hands_feet_hips'
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_handsfeethips'
    launch_experiment(name)

    # all joints
    cfg = default_cfg
    cfg.model.in_params.num_joints = 14
    cfg.dataset.joints = 'all'
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_allJoints'
    launch_experiment(name)

    # only position
    cfg = default_cfg
    cfg.model.in_features =  'kinematics_pos'
    cfg.model.in_params.joint_in_channels = 2
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_posonly'
    launch_experiment(name)

    # only velocity
    cfg = default_cfg
    cfg.model.in_features =  'kinematics_vel_only'
    cfg.model.in_params.joint_in_channels = 2
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_velonly'
    launch_experiment(name)

    # no oversampling
    cfg = default_cfg
    cfg.dataset.sampling.enable = False
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_noOversampling'
    launch_experiment(name)

    # weigthed loss
    cfg = default_cfg
    cfg.hparams.criterion.name = 'scripts.FM_classification.loss:WeightedCrossEntropyLoss'
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_weigthedLoss'
    launch_experiment(name)

    # only minority augmentation
    cfg = default_cfg
    cfg.dataset.transform.params.class_agnostic = False
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_minAugment'
    launch_experiment(name)

    # no augmentation
    cfg = default_cfg
    cfg.dataset.transform.enable = False
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_noAugment'
    launch_experiment(name)

    # sequence sample overlap 250
    cfg = default_cfg
    cfg.dataset.params_clips.max_overlap = 250
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_seqOverlap250'
    launch_experiment(name)

    # sequence sample overlap 500
    cfg = default_cfg
    cfg.dataset.params_clips.max_overlap = 500
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_seqOverlap500'
    launch_experiment(name)

    # clip len 750
    cfg = default_cfg
    cfg.model.in_params.clip_len = 750
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_clipLen750'
    launch_experiment(name)

    # clip 500 overlap 250
    cfg = default_cfg
    cfg.model.in_params.clip_len = 500
    cfg.model.in_params.clip_overlap = 250
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_clip500overlap250'
    launch_experiment(name)

    # clip 500 overlap 375
    cfg = default_cfg
    cfg.model.in_params.clip_len = 500
    cfg.model.in_params.clip_overlap = 375
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_clip500overlap250'
    launch_experiment(name)

    # clip overlap 125
    cfg = default_cfg
    cfg.model.in_params.clip_overlap = 125
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_clipOverlap125'
    launch_experiment(name)

    # clip overlap 250
    cfg = default_cfg
    cfg.model.in_params.clip_overlap = 250
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_clipOverlap250'
    launch_experiment(name)

    # 3class
    cfg = default_cfg
    cfg.model.in_params.num_classes = 3
    cfg.dataset.mapping = {1: 0, 4: 1, 12: 2}
    OmegaConf.save(cfg, cfg_path)
    name = 'FM_cls_clipOverlap125'
    launch_experiment(name)


if __name__ == "__main__":
    # Load the configuration file
    default_cfg = OmegaConf.load("config/train_TF.yaml")
    main(default_cfg)