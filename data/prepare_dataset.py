import argparse
import json
import torch
from pathlib import Path
from torchvision.transforms import v2

from data.bibnet_dataset import BibNetDataset
from data.transform.transforms import ResizeWithPadding


def main():
    parser = argparse.ArgumentParser(description='Prepare BibNet dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    stats_file = data_dir / "dataset" / "dataset_stats.json"

    # Create datasets (h5py files) for train, valid, and test
    # and calculate statistics for (later) normalization
    print("\n========================================")
    print("Creating h5 datasets...")
    print("========================================")

    for mode in ['train', 'valid', 'test']:
        print(f"Building {mode} dataset...")

        BibNetDataset(
            data_dir=data_dir,
            mode=mode,
            transform=v2.Compose([
                ResizeWithPadding((512, 512)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ]),
            force_reload=True
        )
        if mode == 'train':
            # stats will be calculated on training set
            mean, std = BibNetDataset.calculate_dataset_stats(data_dir=data_dir, mode='train')

            stats = {
                "mean": mean.tolist(),
                "std": std.tolist()
            }
            with open(stats_file, 'w') as f:
                json.dump(stats, f)
            print(f"Statistics saved to {stats_file}")

        print(f"Completed {mode} dataset creation.")

    print("\nAll datasets have been prepared successfully!")


if __name__ == '__main__':
    main()
