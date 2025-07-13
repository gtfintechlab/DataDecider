#!/usr/bin/env python3
"""Utilities for creating subsamples of FinPile datasets for testing."""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm


def create_subsample(
    source_prefix: str,
    target_prefix: str,
    num_chunks: int,
    chunk_size: int = 4096,
    token_dtype: np.dtype = np.uint16,
    start_chunk: int = 0,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Create a subsample of a FinPile dataset.

    Args:
        source_prefix: Path to source dataset (without .bin/.idx extension)
        target_prefix: Path to target dataset (without .bin/.idx extension)
        num_chunks: Number of chunks to extract
        chunk_size: Size of each chunk in tokens
        token_dtype: Data type for tokens
        start_chunk: Starting chunk index (for sampling from different parts)
        verbose: Whether to print progress

    Returns:
        Tuple of (total_tokens_copied, total_docs_in_range)
    """
    source_bin = f"{source_prefix}.bin"
    source_idx = f"{source_prefix}.idx"
    target_bin = f"{target_prefix}.bin"
    target_idx = f"{target_prefix}.idx"

    if verbose:
        print(f"Creating subsample from {source_prefix}")
        print(f"Target: {target_prefix}")
        print(f"Chunks: {num_chunks}, Size: {chunk_size}, Start: {start_chunk}")

    # Ensure target directory exists
    os.makedirs(Path(target_prefix).parent, exist_ok=True)

    # Calculate total tokens and docs in source
    source_bin_size = os.path.getsize(source_bin)
    source_total_tokens = source_bin_size // np.dtype(token_dtype).itemsize
    source_idx_size = os.path.getsize(source_idx)
    source_total_docs = source_idx_size // np.dtype(np.uint64).itemsize

    if verbose:
        print(f"Source dataset: {source_total_tokens:,} tokens, {source_total_docs:,} docs")

    # Calculate token range to extract
    start_token = start_chunk * chunk_size
    end_token = min(start_token + (num_chunks * chunk_size), source_total_tokens)
    actual_tokens = end_token - start_token

    if verbose:
        print(f"Extracting tokens {start_token:,} to {end_token:,} ({actual_tokens:,} tokens)")

    # Copy token data
    if verbose:
        print("Copying token data...")

    source_tokens = np.memmap(source_bin, dtype=token_dtype, mode="r")
    target_tokens = np.memmap(target_bin, dtype=token_dtype, mode="w+", shape=(actual_tokens,))

    # Copy in chunks to avoid memory issues
    copy_chunk_size = 1_000_000  # 1M tokens at a time
    for i in tqdm(range(0, actual_tokens, copy_chunk_size), disable=not verbose, desc="Copying tokens"):
        chunk_start = start_token + i
        chunk_end = min(chunk_start + copy_chunk_size, end_token)
        local_end = i + (chunk_end - chunk_start)
        target_tokens[i:local_end] = source_tokens[chunk_start:chunk_end]

    # Flush and close
    del target_tokens
    del source_tokens

    # Now handle the document offsets
    if verbose:
        print("Processing document offsets...")

    source_offsets = np.memmap(source_idx, dtype=np.uint64, mode="r")

    # Find documents that start within our token range
    valid_docs = []
    adjusted_offsets = []

    for doc_idx in tqdm(range(source_total_docs), disable=not verbose, desc="Processing docs"):
        doc_start = source_offsets[doc_idx]

        # Check if document starts within our range
        if start_token <= doc_start < end_token:
            valid_docs.append(doc_idx)
            # Adjust offset to be relative to our new start
            adjusted_offset = doc_start - start_token
            adjusted_offsets.append(adjusted_offset)

    # Write new index file
    if adjusted_offsets:
        target_offsets = np.array(adjusted_offsets, dtype=np.uint64)
        target_offsets.tofile(target_idx)
        docs_copied = len(adjusted_offsets)
    else:
        # Create empty index file
        np.array([], dtype=np.uint64).tofile(target_idx)
        docs_copied = 0

    del source_offsets

    if verbose:
        print("Subsample created successfully!")
        print(f"  Tokens: {actual_tokens:,}")
        print(f"  Documents: {docs_copied:,}")
        print(f"  Files: {target_bin}, {target_idx}")

    return actual_tokens, docs_copied


def create_test_subsamples(source_prefix: str, output_dir: str = "data/finpile_subsamples", chunk_size: int = 4096):
    """
    Create a set of test subsamples of different sizes.

    Args:
        source_prefix: Path to source dataset
        output_dir: Directory to store subsamples
        chunk_size: Chunk size for the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define subsample sizes (in chunks)
    subsample_configs = [
        ("tiny", 10),  # ~40K tokens
        ("small", 100),  # ~400K tokens
        ("medium", 1000),  # ~4M tokens
        ("large", 10000),  # ~40M tokens
        ("xlarge", 50000),  # ~200M tokens
    ]

    print("Creating test subsamples from finpile dataset")
    print("=" * 60)

    for name, num_chunks in subsample_configs:
        target_prefix = output_path / f"finpile_{name}"

        print(f"\nCreating {name} subsample ({num_chunks:,} chunks)...")
        try:
            total_tokens, total_docs = create_subsample(
                source_prefix=source_prefix,
                target_prefix=str(target_prefix),
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                start_chunk=0,  # Always start from beginning for consistency
                verbose=False,
            )

            print(f"✓ {name}: {total_tokens:,} tokens, {total_docs:,} docs")

        except Exception as e:
            print(f"✗ {name}: Failed - {e}")

    print(f"\nAll subsamples created in: {output_path}")


def verify_subsample(prefix: str, expected_chunks: int = None, chunk_size: int = 4096):
    """
    Verify that a subsample was created correctly.

    Args:
        prefix: Path to subsample (without extension)
        expected_chunks: Expected number of chunks (optional)
        chunk_size: Chunk size
    """
    bin_file = f"{prefix}.bin"
    idx_file = f"{prefix}.idx"

    print(f"Verifying subsample: {prefix}")

    # Check files exist
    if not os.path.exists(bin_file):
        print(f"✗ Binary file missing: {bin_file}")
        return False

    if not os.path.exists(idx_file):
        print(f"✗ Index file missing: {idx_file}")
        return False

    # Check sizes
    bin_size = os.path.getsize(bin_file)
    idx_size = os.path.getsize(idx_file)

    total_tokens = bin_size // 2  # uint16
    total_docs = idx_size // 8  # uint64
    actual_chunks = (total_tokens + chunk_size - 1) // chunk_size

    print(f"  Tokens: {total_tokens:,}")
    print(f"  Documents: {total_docs:,}")
    print(f"  Chunks: {actual_chunks:,}")

    if expected_chunks and actual_chunks != expected_chunks:
        print(f"  ⚠️  Expected {expected_chunks:,} chunks, got {actual_chunks:,}")

    # Test loading
    try:
        from FinPileTokenizers.fsiltok.utils.data import DocumentTapeDataset

        dataset = DocumentTapeDataset(prefix, chunk_size=chunk_size)
        sample = dataset[0]
        print("  ✓ Loading test passed")
        return True
    except Exception as e:
        print(f"  ✗ Loading test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create FinPile dataset subsamples")
    parser.add_argument(
        "--source",
        default="/storage/coda1/p-schava6/0/shared/finpile/datamixes/0fp-100dolma",
        help="Source dataset prefix",
    )
    parser.add_argument("--output-dir", default="data/finpile_subsamples", help="Output directory for subsamples")
    parser.add_argument("--verify", action="store_true", help="Verify existing subsamples instead of creating new ones")

    args = parser.parse_args()

    if args.verify:
        # Verify existing subsamples
        output_path = Path(args.output_dir)
        if not output_path.exists():
            print(f"Output directory doesn't exist: {output_path}")
            return

        subsample_names = ["tiny", "small", "medium", "large", "xlarge"]
        for name in subsample_names:
            prefix = output_path / f"finpile_{name}"
            if prefix.with_suffix(".bin").exists():
                verify_subsample(str(prefix))
            else:
                print(f"Subsample not found: {name}")
    else:
        # Create subsamples
        create_test_subsamples(source_prefix=args.source, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
