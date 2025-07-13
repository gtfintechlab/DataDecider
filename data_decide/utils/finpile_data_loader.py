"""Simple data loader for FinPileTokenizers."""

import numpy as np
import torch
from fsiltok.utils.data import DocumentTapeDataset
from torch.utils.data import DataLoader


class SimpleTapeDataset(DocumentTapeDataset):
    """Simple wrapper to make DocumentTapeDataset work with PyTorch with proper resource management."""

    def __init__(self, prefix, chunk_size=4096, token_dtype=np.uint16, eod_token_id=None):
        """Initialize the dataset.

        Args:
            prefix: Path to dataset (without .bin/.idx extension)
            chunk_size: Size of each chunk in tokens
            token_dtype: Data type for tokens (uint16 for FinPile)
            eod_token_id: End of document token ID (None for no special handling)
        """
        super().__init__(prefix=prefix, token_dtype=token_dtype, chunk_size=chunk_size, eod_token_id=eod_token_id)
        self._is_closed = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()

    def close(self):
        """Explicitly close and clean up memory-mapped files."""
        if not self._is_closed:
            if hasattr(self, "_handle") and self._handle is not None:
                # For numpy memmap, setting to None and deleting helps with cleanup
                del self._handle
                self._handle = None

            if hasattr(self, "_offsets") and self._offsets is not None:
                del self._offsets
                self._offsets = None

            self._is_closed = True

    def __del__(self):
        """Destructor to ensure cleanup happens."""
        if hasattr(self, "_is_closed") and not self._is_closed:
            self.close()

    def __getitem__(self, idx):
        """Get a training sample with early validation and detailed error messages."""
        # Early validation before any operations
        from .validation import EarlyValidator, validate_dataset_bounds

        with EarlyValidator("dataset_access") as validator:
            # Check dataset state first
            validator.add_validation(lambda: self._validate_dataset_state())

            # Validate index bounds with detailed error messages
            validator.add_validation(
                lambda: validate_dataset_bounds(idx, self._total_chunks, f"SimpleTapeDataset({self.prefix})")
            )

    def _validate_dataset_state(self):
        """Validate that the dataset is in a valid state for access."""
        from .validation import DataValidationError

        if self._is_closed:
            raise DataValidationError(
                "Cannot access data from a closed dataset",
                suggestions=[
                    "Re-open the dataset or create a new instance",
                    "Use dataset within a 'with' statement for automatic management",
                    "Check if dataset.close() was called prematurely",
                ],
            )

        if not hasattr(self, "_total_chunks") or self._total_chunks <= 0:
            raise DataValidationError(
                "Dataset not properly initialized - missing or invalid chunk count",
                field="_total_chunks",
                value=getattr(self, "_total_chunks", "missing"),
                suggestions=[
                    "Ensure dataset was properly initialized with valid data files",
                    f"Check that .idx file exists: {self.prefix}.idx",
                    f"Check that .bin file exists: {self.prefix}.bin",
                    "Verify file permissions and accessibility",
                ],
            )

        # Fixed: Check if handle is None properly (bug in original DocumentTapeDataset)
        if self._handle is None:
            try:
                self._handle = np.memmap(
                    f"{self.prefix}.bin", dtype=self._token_dtype, mode="r", shape=(self._total_tokens,)
                )
            except (OSError, ValueError) as e:
                raise RuntimeError(f"Failed to open data file '{self.prefix}.bin': {e}")

        if self._offsets is None:
            try:
                self._offsets = np.memmap(
                    f"{self.prefix}.idx", dtype=self._offsets_dtype, mode="r", shape=(self._total_docs,)
                )
            except (OSError, ValueError) as e:
                raise RuntimeError(f"Failed to open index file '{self.prefix}.idx': {e}")

        id_start = self._chunk_size * idx
        id_end = min(id_start + self._chunk_size, self._total_tokens)

        tokens = self._handle[id_start:id_end]
        if len(tokens) == 0:
            tokens = np.array([], dtype=self._token_dtype)

        # Handle packed mode if eod_token_id is set
        if self._eod_token_id is not None:
            # Note: Packed sequence handling is implemented in PackedTapeDataset class
            # This simple version treats all tokens equally
            pass

        # Convert to torch tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class PackedTapeDataset(DocumentTapeDataset):
    """Dataset that properly handles packed sequences with position IDs."""

    def __init__(self, prefix, chunk_size=4096, token_dtype=np.uint16, eod_token_id=50256):
        """Initialize with end-of-document token handling.

        Args:
            prefix: Path to dataset (without .bin/.idx extension)
            chunk_size: Size of each chunk in tokens
            token_dtype: Data type for tokens (uint16 for FinPile)
            eod_token_id: End of document token ID (e.g., 50256 for GPT-NeoX)
        """
        super().__init__(prefix=prefix, token_dtype=token_dtype, chunk_size=chunk_size, eod_token_id=eod_token_id)

    def __getitem__(self, idx):
        """Get a training sample with proper packed sequence handling."""
        data = super().__getitem__(idx)

        if isinstance(data, dict):
            # Use the packed features
            tokens = torch.tensor(data["tokens"], dtype=torch.long)
            position_ids = torch.tensor(data["position_ids"], dtype=torch.long)
            # Note: mask is 2D causal mask, we'll create 1D attention mask
            attention_mask = torch.ones_like(tokens, dtype=torch.long)

            return {
                "input_ids": tokens,
                "labels": tokens.clone(),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "packed_mask": torch.tensor(data["mask"], dtype=torch.bool),
            }
        else:
            # Fallback to simple handling
            tokens = torch.tensor(data, dtype=torch.long)
            return {
                "input_ids": tokens,
                "labels": tokens.clone(),
                "attention_mask": torch.ones_like(tokens, dtype=torch.long),
            }


def load_finpile_dataset(prefix, batch_size=32, packed=False, eod_token_id=None, **kwargs):
    """Load a FinPile dataset.

    Args:
        prefix: Path to dataset (without .bin/.idx extension)
        batch_size: Batch size
        packed: Whether to use packed sequences with document boundaries
        eod_token_id: End of document token ID (required for packed=True)
        **kwargs: Additional args for DataLoader

    Returns:
        PyTorch DataLoader
    """
    if packed:
        if eod_token_id is None:
            raise ValueError("eod_token_id is required when packed=True")
        dataset = PackedTapeDataset(prefix, eod_token_id=eod_token_id)
    else:
        dataset = SimpleTapeDataset(prefix, eod_token_id=eod_token_id)

    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def create_data_collator_for_finpile():
    """Create a data collator for FinPile data that handles variable length sequences."""

    def collate_fn(batch):
        """Collate function for FinPile data."""
        # All sequences should be the same length (chunk_size), so simple stacking works
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Add position_ids and packed_mask if present
        if "position_ids" in batch[0]:
            result["position_ids"] = torch.stack([item["position_ids"] for item in batch])

        if "packed_mask" in batch[0]:
            result["packed_mask"] = torch.stack([item["packed_mask"] for item in batch])

        return result

    return collate_fn
