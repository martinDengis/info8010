import argparse
from utils.wandb_integration import main as launcher

if __name__ == "__main__":
    # ---------- Launcher ----------
    parser = argparse.ArgumentParser(description='Launch training with specified model type')
    parser.add_argument('--model_type', type=str, default=None, help='Model type to use: "bibnet" or "yolov1"')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep instead of single experiment')
    args = parser.parse_args()

    # Pass model_type to launcher
    launcher(model_type=args.model_type)

    # ---------- Local Testing ----------
    # from tools.train_net import train
    # from config.defaults import get_cfg_defaults
    # cfg = get_cfg_defaults()
    # print('Launching training with default config...')
    # train(cfg)