#!/usr/bin/env python3
"""Debug the FinPile data loader issue."""

import sys
import traceback
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from data_decide.utils.finpile_data_loader import SimpleTapeDataset


def debug_single_sample():
    """Debug loading a single sample."""
    print("Testing single sample load...")

    prefix = "data/finpile_subsamples/finpile_tiny"

    try:
        dataset = SimpleTapeDataset(prefix, chunk_size=4096)
        print(f"Dataset created, length: {len(dataset)}")

        # Try to load sample 0
        print("Loading sample 0...")
        sample0 = dataset[0]
        print("Sample 0 loaded successfully")
        print(f"  Type: {type(sample0)}")
        print(f"  Keys: {sample0.keys()}")

        # Try to load sample 1
        print("Loading sample 1...")
        sample1 = dataset[1]
        print("Sample 1 loaded successfully")

        return True

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def debug_iteration():
    """Debug iterating through the dataset."""
    print("\nTesting dataset iteration...")

    prefix = "data/finpile_subsamples/finpile_tiny"

    try:
        dataset = SimpleTapeDataset(prefix, chunk_size=4096)

        # Try to iterate
        for i, sample in enumerate(dataset):
            print(f"Loaded sample {i}")
            if i >= 2:  # Just test first 3
                break

        return True

    except Exception as e:
        print(f"Error during iteration: {e}")
        traceback.print_exc()
        return False


def debug_raw_documenttape():
    """Debug the raw DocumentTapeDataset."""
    print("\nTesting raw DocumentTapeDataset...")

    prefix = "data/finpile_subsamples/finpile_tiny"

    try:
        from FinPileTokenizers.fsiltok.utils.data import DocumentTapeDataset

        dataset = DocumentTapeDataset(prefix, chunk_size=4096)
        print(f"Raw dataset length: {len(dataset)}")

        # Test multiple samples
        for i in range(min(3, len(dataset))):
            print(f"Loading raw sample {i}...")
            sample = dataset[i]
            print(f"  Type: {type(sample)}")
            if hasattr(sample, "shape"):
                print(f"  Shape: {sample.shape}")
            elif isinstance(sample, dict):
                print(f"  Keys: {sample.keys()}")

        return True

    except Exception as e:
        print(f"Error with raw dataset: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Debugging FinPile Data Loader")
    print("=" * 40)

    debug_raw_documenttape()
    debug_single_sample()
    debug_iteration()
