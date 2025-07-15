#!/usr/bin/env python3
"""Create just small subsamples for immediate testing."""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from data_decide.utils.subsample_finpile import create_subsample


def main():
    source_prefix = "/storage/coda1/p-schava6/0/shared/finpile/datamixes/0fp-100dolma"
    output_dir = Path("data/finpile_subsamples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create only small subsamples for immediate testing
    small_configs = [
        ("tiny", 10),  # ~40K tokens
        ("small", 100),  # ~400K tokens
        ("medium", 1000),  # ~4M tokens
    ]

    print("Creating small test subsamples")
    print("=" * 40)

    for name, num_chunks in small_configs:
        target_prefix = output_dir / f"finpile_{name}"

        print(f"\nCreating {name} subsample ({num_chunks:,} chunks)...")
        try:
            total_tokens, total_docs = create_subsample(
                source_prefix=source_prefix,
                target_prefix=str(target_prefix),
                num_chunks=num_chunks,
                chunk_size=4096,
                start_chunk=0,
                verbose=True,
            )

            print(f"✓ {name}: {total_tokens:,} tokens, {total_docs:,} docs")

        except Exception as e:
            print(f"✗ {name}: Failed - {e}")


if __name__ == "__main__":
    main()
