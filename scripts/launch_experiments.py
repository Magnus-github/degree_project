import sys
sys.path.append(".")
from scripts.tf_configs import TF_configs
from scripts.smnn_configs import SMNN_configs
from scripts.tcn_configs import TCN_configs
from scripts.gcntf_configs import GCNTF_configs


if __name__ == "__main__":
    # Load the configuration file
    # cfg_path = "config/train_TF.yaml"
    # TF_configs(cfg_path)
    # cfg_path = "config/train_SMNN.yaml"
    # SMNN_configs(cfg_path)
    # cfg_path = "config/train_TCN.yaml"
    # TCN_configs(cfg_path)
    cfg_path = "config/train_GCNTF.yaml"
    GCNTF_configs(cfg_path)