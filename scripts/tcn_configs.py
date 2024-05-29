from omegaconf import OmegaConf
import os
import subprocess

def launch_experiment(name: str, cfg_path: str):
    cmd = ['sbatch', '--export', f'CONFIG={cfg_path},PROJECT_NAME={name}', 'train_TCN']
    result = subprocess.Popen(cmd).wait()
    if result != 0:
        print(f"Error launching experiment {name}")
    return


def TCN_configs(cfg_path: str):
    cfg_folder = os.path.dirname(cfg_path)
    # Load the configuration file
    default_cfg = OmegaConf.load(cfg_path)
    name = 'FM_cls_base'
    launch_experiment(name, cfg_path)

    # only hands
    cfg = default_cfg.copy()
    cfg.model.in_params.num_joints = 2
    cfg.dataset.joints = 'hands'
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_handsonly')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_handsonly'
    launch_experiment(name, cfg_save_path)

    # only feet
    cfg = default_cfg.copy()
    cfg.model.in_params.num_joints = 2
    cfg.dataset.joints = 'feet'
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_feetonly')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_feetonly'
    launch_experiment(name, cfg_save_path)

    # only hips
    cfg = default_cfg.copy()
    cfg.model.in_params.num_joints = 2
    cfg.dataset.joints = 'hips'
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_hipsonly')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_hipsonly'
    launch_experiment(name, cfg_save_path)

    # hands_feet_hips
    cfg = default_cfg.copy()
    cfg.model.in_params.num_joints = 6
    cfg.dataset.joints = 'hands_feet_hips'
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_handsfeethips')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_handsfeethips'
    launch_experiment(name, cfg_save_path)

    # all joints
    cfg = default_cfg.copy()
    cfg.model.in_params.num_joints = 14
    cfg.dataset.joints = 'all'
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_allJoints')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_allJoints'
    launch_experiment(name, cfg_save_path)

    # only position
    cfg = default_cfg.copy()
    cfg.model.in_features =  'kinematics_pos'
    cfg.model.in_params.joint_in_channels = 2
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_posonly')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_posonly'
    launch_experiment(name, cfg_save_path)

    # only velocity
    cfg = default_cfg.copy()
    cfg.model.in_features =  'kinematics_vel_only'
    cfg.model.in_params.joint_in_channels = 2
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_velonly')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_velonly'
    launch_experiment(name, cfg_save_path)

    # no oversampling
    cfg = default_cfg.copy()
    cfg.dataset.sampling.enable = False
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_noOversampling')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_noOversampling'
    launch_experiment(name, cfg_save_path)

    # weigthed loss
    cfg = default_cfg.copy()
    cfg.hparams.criterion.name = 'scripts.FM_classification.loss:WeightedCrossEntropyLoss'
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_weigthedLoss')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_weigthedLoss'
    launch_experiment(name, cfg_save_path)

    # only minority augmentation
    cfg = default_cfg.copy()
    cfg.dataset.transform.params.class_agnostic = False
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_minAugment')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_minAugment'
    launch_experiment(name, cfg_save_path)

    # no augmentation
    cfg = default_cfg.copy()
    cfg.dataset.transform.enable = False
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_noAugment')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_noAugment'
    launch_experiment(name, cfg_save_path)

    # sequence sample overlap 250
    cfg = default_cfg.copy()
    cfg.dataset.params_clips.max_overlap = 250
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_seqOverlap250')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_seqOverlap250'
    launch_experiment(name, cfg_save_path)

    # sequence sample overlap 500
    cfg = default_cfg.copy()
    cfg.dataset.params_clips.max_overlap = 500
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_seqOverlap500')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_seqOverlap500'
    launch_experiment(name, cfg_save_path)

    # clip len 750
    cfg = default_cfg.copy()
    cfg.model.in_params.clip_len = 750
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_clipLen750')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_clipLen750'
    launch_experiment(name, cfg_save_path)

    # clip 500 overlap 250
    cfg = default_cfg.copy()
    cfg.model.in_params.clip_len = 500
    cfg.model.in_params.clip_overlap = 250
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_clip500overlap250')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_clip500overlap250'
    launch_experiment(name, cfg_save_path)

    # clip 500 overlap 375
    cfg = default_cfg.copy()
    cfg.model.in_params.clip_len = 500
    cfg.model.in_params.clip_overlap = 375
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_clip500overlap375')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_clip500overlap375'
    launch_experiment(name, cfg_save_path)

    # clip overlap 125
    cfg = default_cfg.copy()
    cfg.model.in_params.clip_overlap = 125
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_clipOverlap125')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_clipOverlap125'
    launch_experiment(name, cfg_save_path)

    # clip overlap 250
    cfg = default_cfg.copy()
    cfg.model.in_params.clip_overlap = 250
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_clipOverlap250')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_clipOverlap250'
    launch_experiment(name, cfg_save_path)

    # 3class
    cfg = default_cfg.copy()
    cfg.model.in_params.num_classes = 3
    cfg.dataset.mapping = {1: 0, 4: 1, 12: 2}
    cfg_dir = os.path.join(cfg_folder, 'FM_cls_3class_base')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_save_path = os.path.join(cfg_dir, 'train_TCN.yaml')
    OmegaConf.save(cfg, cfg_save_path)
    name = 'FM_cls_3class_base'
    launch_experiment(name, cfg_save_path)
