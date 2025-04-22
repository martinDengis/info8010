import argparse
import json
import os
from pathlib import Path
from data.bibnet_dataset import BibNetDataset
from data.transform.build import build_train_transforms, build_test_transforms


def main():
    parser = argparse.ArgumentParser(description='Prepare BibNet dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force regeneration of H5 files')
    parser.add_argument('--calculate_stats', action='store_true',
                        help='Calculate dataset statistics (mean, std) and save to JSON file')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    stats_file = data_dir / "dataset" / "dataset_stats.json"

    # Phase 1: Calculate dataset statistics if needed
    if args.calculate_stats or not os.path.exists(stats_file):
        print("\n========================================")
        print("Calculating dataset statistics...")
        print("========================================")
        # stats will be calculated on training set w/ only ToTensor transform
        mean, std = BibNetDataset.calculate_dataset_stats(
            data_dir, mode='train')

        stats = {
            "mean": mean.tolist(),
            "std": std.tolist()
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        print(f"Statistics saved to {stats_file}")

    # Phase 2: Create datasets with transforms
    print("\n========================================")
    print("Creating datasets with transforms...")
    print("========================================")

    # The transforms will automatically use the dataset stats if available
    for mode in ['train', 'valid', 'test']:
        print(f"Building {mode} dataset...")
        if mode == 'train':
            transform = build_train_transforms(data_dir=data_dir)
        else:
            transform = build_test_transforms(data_dir=data_dir)

        BibNetDataset(
            data_dir=data_dir,
            mode=mode,
            transform=transform,
            force_reload=args.force_reload
        )
        print(f"Completed {mode} dataset creation.")

    print("\nAll datasets have been prepared successfully!")


if __name__ == '__main__':
    main()
