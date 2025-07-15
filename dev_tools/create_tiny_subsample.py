#!/usr/bin/env python3
"""Create a tiny subsample quickly for immediate testing."""

from pathlib import Path

import numpy as np


def create_tiny_subsample():
    """Create a tiny subsample with just the first 40KB of data."""
    source_prefix = "/storage/coda1/p-schava6/0/shared/finpile/datamixes/0fp-100dolma"

    # Create output directory
    output_dir = Path("data/finpile_subsamples")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_prefix = output_dir / "finpile_tiny"
    target_bin = f"{target_prefix}.bin"
    target_idx = f"{target_prefix}.idx"

    print("Creating tiny subsample (first 20K tokens)...")

    # Read first 20K tokens (40KB since uint16)
    num_tokens = 20480  # 20K tokens = 5 chunks of 4096

    print("Reading source data...")
    source_bin = f"{source_prefix}.bin"
    source_tokens = np.memmap(source_bin, dtype=np.uint16, mode="r")

    # Copy first 20K tokens
    print("Copying tokens...")
    target_data = source_tokens[:num_tokens].copy()
    target_data.tofile(target_bin)

    print("Creating document index...")
    # Read source index to find docs within our range
    source_idx = f"{source_prefix}.idx"
    source_offsets = np.memmap(source_idx, dtype=np.uint64, mode="r")

    # Find documents that start within first 20K tokens
    valid_offsets = []
    for offset in source_offsets:
        if offset < num_tokens:
            valid_offsets.append(offset)
        else:
            break  # Offsets are sorted, so we can stop

    # Write new index
    if valid_offsets:
        np.array(valid_offsets, dtype=np.uint64).tofile(target_idx)
    else:
        # Create minimal index with just offset 0
        np.array([0], dtype=np.uint64).tofile(target_idx)

    print("Tiny subsample created:")
    print(f"  Tokens: {num_tokens:,}")
    print(f"  Documents: {len(valid_offsets):,}")
    print(f"  Files: {target_bin}, {target_idx}")

    # Test loading
    print("Testing load...")
    try:
        import sys

        sys.path.append(".")
        from fsiltok.utils.data import DocumentTapeDataset

        dataset = DocumentTapeDataset(str(target_prefix), chunk_size=4096)
        print(f"  Dataset length: {len(dataset):,} chunks")

        sample = dataset[0]
        if isinstance(sample, np.ndarray):
            print(f"  Sample shape: {sample.shape}")
            print(f"  First 5 tokens: {sample[:5]}")

        print("✓ Loading test successful!")

    except Exception as e:
        print(f"✗ Loading test failed: {e}")


if __name__ == "__main__":
    create_tiny_subsample()
