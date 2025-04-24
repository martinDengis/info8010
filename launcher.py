import argparse
from utils.wandb_integration import main as launcher

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch training with specified model type')
    parser.add_argument('--model_type', type=str, default=None,
                       help='Model type to use: bibnet or bibc3net')
    args = parser.parse_args()

    # Pass the model_type to the launcher
    launcher(model_type=args.model_type)