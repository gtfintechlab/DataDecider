"""Simple data loader for FinPileTokenizers."""

import torch
from torch.utils.data import DataLoader
from FinPileTokenizers.fsiltok.utils.data import DocumentTapeDataset


class SimpleTapeDataset(DocumentTapeDataset):
    """Simple wrapper to make DocumentTapeDataset work with PyTorch."""
    
    def __getitem__(self, idx):
        tokens = super().__getitem__(idx)
        
        # If it's a dict (packed mode), just get tokens
        if isinstance(tokens, dict):
            tokens = tokens["tokens"]
            
        # Convert to torch tensor
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.ones(len(tokens), dtype=torch.long),
        }


def load_finpile_dataset(prefix, batch_size=32, **kwargs):
    """Load a FinPile dataset.
    
    Args:
        prefix: Path to dataset (without .bin/.idx extension)
        batch_size: Batch size
        **kwargs: Additional args for DataLoader
        
    Returns:
        PyTorch DataLoader
    """
    dataset = SimpleTapeDataset(prefix)
    return DataLoader(dataset, batch_size=batch_size, **kwargs)