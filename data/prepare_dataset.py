import argparse
import json
import os
from pathlib import Path
from build import build_dataset
from bibnet import BibNetDataset
import torchvision.transforms as transforms

def main():
    parser = argparse.ArgumentParser(description='Prepare BibNet dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force regeneration of H5 files')
    parser.add_argument('--calculate_stats', action='store_true',
                        help='Calculate dataset statistics')
    parser.add_argument('--apply_normalization', action='store_true',
                        help='Apply normalization using calculated statistics')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    stats_file = data_dir / "dataset_stats.json"

    # 1: Create initial datasets without normalization
    if args.force_reload or not all(os.path.exists(data_dir / f"bib_{mode}.h5")
                                    for mode in ['train', 'valid', 'test']):
        print("\n========================================")
        print("1: Creating initial datasets without normalization...")
        print("========================================")
        for mode in ['train', 'valid', 'test']:
            build_dataset(data_dir, mode=mode, force_reload=True, transform=None)

    # 2: Calculate dataset statistics if requested
    if args.calculate_stats or (args.apply_normalization and not os.path.exists(stats_file)):
        print("\n========================================")
        print("2: Calculating dataset statistics...")
        print("========================================")
        # Calculate statistics on the training set
        mean, std = BibNetDataset.calculate_dataset_stats(data_dir, mode='train')

        # Save statistics to file
        stats = {
            "mean": mean.tolist(),
            "std": std.tolist()
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        print(f"Statistics saved to {stats_file}")

    # 3: Recreate datasets with normalization if requested
    if args.apply_normalization:
        print("\n========================================")
        print("3: Recreating datasets with normalization...")
        print("========================================")

        # Load statistics
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Create normalize transform
        normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=stats["mean"], std=stats["std"])
        ])

        # Rebuild datasets with normalization
        for mode in ['train', 'valid', 'test']:
            build_dataset(data_dir, mode=mode, force_reload=True,
                         transform=normalize_transform)

    print("All datasets have been prepared!")

if __name__ == '__main__':
    main()
