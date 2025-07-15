#!/usr/bin/env python3
"""Quick test to verify finpile data loading basics."""

import sys
from pathlib import Path

import numpy as np

# Add the project root and submodule to Python path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "FinPileTokenizers"))


def quick_file_check():
    """Quick check of file format."""
    prefix = "/storage/coda1/p-schava6/0/shared/finpile/datamixes/0fp-100dolma"

    print("File format check:")
    print(f"Bin file exists: {Path(f'{prefix}.bin').exists()}")
    print(f"Idx file exists: {Path(f'{prefix}.idx').exists()}")

    # Check first few bytes of idx file
    with open(f"{prefix}.idx", "rb") as f:
        first_bytes = f.read(32)  # Read first 32 bytes (4 uint64 values)
        offsets = np.frombuffer(first_bytes, dtype=np.uint64)
        print(f"First 4 document offsets: {offsets}")

    # Check token dtype by looking at bin file size
    bin_size = Path(f"{prefix}.bin").stat().st_size
    if bin_size % 2 == 0:  # uint16
        total_tokens_uint16 = bin_size // 2
        print(f"If uint16: {total_tokens_uint16:,} tokens")
    if bin_size % 4 == 0:  # uint32
        total_tokens_uint32 = bin_size // 4
        print(f"If uint32: {total_tokens_uint32:,} tokens")


def quick_memmap_test():
    """Quick memory map test."""
    prefix = "/storage/coda1/p-schava6/0/shared/finpile/datamixes/0fp-100dolma"

    print("\nMemory map test:")
    try:
        # Try to memory map just the first 1000 tokens
        token_data = np.memmap(f"{prefix}.bin", dtype=np.uint16, mode="r", shape=(1000,))
        print(f"First 10 tokens: {token_data[:10]}")
        print(f"Token range in sample: {token_data.min()} to {token_data.max()}")
        print("✓ Memory mapping works")
    except Exception as e:
        print(f"✗ Memory mapping failed: {e}")


def quick_dataset_test():
    """Quick DocumentTapeDataset test."""
    print("\nDocumentTapeDataset test:")
    try:
        from fsiltok.utils.data import DocumentTapeDataset

        prefix = "/storage/coda1/p-schava6/0/shared/finpile/datamixes/0fp-100dolma"

        # Create dataset with small chunk size for faster testing
        dataset = DocumentTapeDataset(
            prefix=prefix,
            token_dtype=np.uint16,
            chunk_size=100,  # Small chunk for testing
            eod_token_id=None,
        )

        print(f"Dataset created. Length: {len(dataset):,}")

        # Try to load just one sample
        sample = dataset[0]
        if isinstance(sample, np.ndarray):
            print(f"Sample 0: shape={sample.shape}, first_5={sample[:5]}")
        else:
            print(f"Sample 0: type={type(sample)}")
        print("✓ Dataset loading works")

    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")


if __name__ == "__main__":
    print("Quick FinPile Data Compatibility Test")
    print("=" * 50)

    quick_file_check()
    quick_memmap_test()
    quick_dataset_test()

    print("\nDone!")
