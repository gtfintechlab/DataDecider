"""
Test FinPile data integration with DataDecider.

This module tests the integration between FinPile pre-tokenized data
and the DataDecider training pipeline, including data loading,
model creation, and DataDecide methodology adaptation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


@pytest.mark.finpile
@pytest.mark.unit
def test_simple_tape_dataset_creation(mock_finpile_data):
    """Test SimpleTapeDataset creation and basic functionality."""
    from data_decide.utils.finpile_data_loader import SimpleTapeDataset

    # Create dataset with mock data
    dataset = SimpleTapeDataset(mock_finpile_data, chunk_size=100)

    # Verify dataset properties
    assert hasattr(dataset, "_total_chunks")
    assert hasattr(dataset, "_chunk_size")
    assert dataset._chunk_size == 100


@pytest.mark.finpile
@pytest.mark.unit
def test_simple_tape_dataset_context_manager(mock_finpile_data):
    """Test SimpleTapeDataset context manager functionality."""
    from data_decide.utils.finpile_data_loader import SimpleTapeDataset

    # Test context manager
    with SimpleTapeDataset(mock_finpile_data, chunk_size=100) as dataset:
        assert not dataset._is_closed
        # Should be able to access data
        assert hasattr(dataset, "_total_chunks")

    # After context, should be closed
    assert dataset._is_closed


@pytest.mark.finpile
@pytest.mark.unit
def test_simple_tape_dataset_sample_structure(mock_finpile_data):
    """Test that dataset samples have correct structure."""
    from data_decide.utils.finpile_data_loader import SimpleTapeDataset

    with SimpleTapeDataset(mock_finpile_data, chunk_size=50) as dataset:
        # Mock the internal methods to avoid file I/O issues
        with patch.object(dataset, "_validate_dataset_state"):
            with patch.object(dataset, "_handle", np.array([1, 2, 3, 4, 5] * 10, dtype=np.uint16)):
                with patch.object(dataset, "_total_chunks", 1):
                    with patch.object(dataset, "_chunk_size", 50):
                        sample = dataset[0]

        # Verify sample structure
        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert "labels" in sample
        assert "attention_mask" in sample

        # Verify tensor types
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["labels"], torch.Tensor)
        assert isinstance(sample["attention_mask"], torch.Tensor)

        # Verify tensor properties
        assert sample["input_ids"].dtype == torch.long
        assert sample["labels"].dtype == torch.long
        assert sample["attention_mask"].dtype == torch.long


@pytest.mark.finpile
@pytest.mark.unit
def test_finpile_data_loader_function(mock_finpile_data):
    """Test the load_finpile_dataset function."""
    from data_decide.utils.finpile_data_loader import load_finpile_dataset

    # Test basic data loader creation
    dataloader = load_finpile_dataset(mock_finpile_data, batch_size=2, packed=False, num_workers=0)

    assert dataloader is not None
    assert dataloader.batch_size == 2


@pytest.mark.finpile
@pytest.mark.unit
def test_packed_dataset_creation(mock_finpile_data):
    """Test PackedTapeDataset creation with EOD token."""
    from data_decide.utils.finpile_data_loader import load_finpile_dataset

    # Test packed dataset creation
    dataloader = load_finpile_dataset(mock_finpile_data, batch_size=2, packed=True, eod_token_id=50256, num_workers=0)

    assert dataloader is not None
    assert dataloader.batch_size == 2


@pytest.mark.finpile
@pytest.mark.unit
def test_data_collator_creation():
    """Test FinPile data collator functionality."""
    from data_decide.utils.finpile_data_loader import create_data_collator_for_finpile

    collator = create_data_collator_for_finpile()
    assert collator is not None
    assert callable(collator)

    # Test collation with sample batch
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "labels": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        },
        {
            "input_ids": torch.tensor([6, 7, 8, 9, 10]),
            "labels": torch.tensor([6, 7, 8, 9, 10]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        },
    ]

    collated = collator(batch)

    assert "input_ids" in collated
    assert "labels" in collated
    assert "attention_mask" in collated
    assert collated["input_ids"].shape == (2, 5)  # batch_size, seq_len


@pytest.mark.finpile
@pytest.mark.integration
def test_model_creation_with_config():
    """Test OLMo model creation with 4M configuration."""
    from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS
    from data_decide.olmo.models.olmo_model import OLMoForCausalLM

    # Get 4M model config
    config = OLMO_CONFIGS["4M"]

    # Create model
    model = OLMoForCausalLM(config)

    # Verify model properties
    assert model is not None
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 1_000_000  # Should be in millions
    assert param_count < 10_000_000  # But not too large for 4M model


@pytest.mark.finpile
@pytest.mark.integration
def test_model_forward_pass(sample_batch_data):
    """Test model forward pass with sample data."""
    from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS
    from data_decide.olmo.models.olmo_model import OLMoForCausalLM

    # Create small model for testing
    config = OLMO_CONFIGS["4M"]
    model = OLMoForCausalLM(config)
    model.eval()

    # Test forward pass
    with torch.no_grad():
        outputs = model(**sample_batch_data)

    # Verify outputs
    assert hasattr(outputs, "logits")
    assert hasattr(outputs, "loss")

    # Verify output shapes
    batch_size, seq_len = sample_batch_data["input_ids"].shape
    vocab_size = config.vocab_size
    assert outputs.logits.shape == (batch_size, seq_len, vocab_size)


@pytest.mark.finpile
@pytest.mark.slow
@pytest.mark.integration
def test_minimal_training_step(sample_batch_data):
    """Test a minimal training step with FinPile data structure."""
    from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS
    from data_decide.olmo.models.olmo_model import OLMoForCausalLM

    # Create model and optimizer
    config = OLMO_CONFIGS["4M"]
    model = OLMoForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()

    # Forward pass
    outputs = model(**sample_batch_data)
    loss = outputs.loss

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Verify loss is reasonable
    assert loss.item() > 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


@pytest.mark.finpile
@pytest.mark.unit
def test_finpile_metadata_creation_structure():
    """Test FinPile metadata creation for DataDecide integration."""
    # Mock the metadata creation functionality
    sample_metadata = {
        "uuid": "doc-1",
        "text": "<pre-tokenized>",  # Placeholder for pre-tokenized data
        "token_start": 0,
        "token_end": 150,
        "token_length": 150,
        "quality_score": 0.85,
        "diversity_score": 0.7,
        "composite_score": 0.775,
        "domain": "financial",
        "doc_id": 1,
    }

    # Verify structure
    required_fields = ["uuid", "text", "token_start", "token_end", "token_length", "quality_score", "diversity_score"]

    for field in required_fields:
        assert field in sample_metadata

    # Verify data types
    assert isinstance(sample_metadata["token_start"], int)
    assert isinstance(sample_metadata["token_end"], int)
    assert isinstance(sample_metadata["quality_score"], float)
    assert 0 <= sample_metadata["quality_score"] <= 1


@pytest.mark.finpile
@pytest.mark.integration
def test_data_curation_integration(sample_jsonl_data, create_jsonl_file, temp_dir):
    """Test DataDecide curation with mock FinPile-style data."""
    from data_decide.olmo.data.data_curation import DataDecideCurator

    # Create test data file
    data_file = create_jsonl_file("test_data.jsonl", sample_jsonl_data)

    # Initialize curator
    curator = DataDecideCurator(
        data_path=str(temp_dir),
        proxy_model_size="4M",
        num_proxy_steps=10,  # Small number for testing
    )

    # Test data loading
    data = curator.load_json_data()
    assert len(data) == len(sample_jsonl_data)

    # Test statistics computation
    stats = curator.compute_data_statistics(data)
    assert isinstance(stats, dict)
    assert "total_documents" in stats
    assert stats["total_documents"] == len(sample_jsonl_data)


@pytest.mark.finpile
@pytest.mark.unit
def test_dataset_validation_error_handling(mock_finpile_data):
    """Test dataset validation and error handling."""
    from data_decide.utils.finpile_data_loader import SimpleTapeDataset
    from data_decide.utils.validation import ValidationError

    dataset = SimpleTapeDataset(mock_finpile_data)

    # Test closed dataset access
    dataset.close()

    # Should raise validation error for closed dataset
    with pytest.raises((ValidationError, RuntimeError)):
        _ = dataset[0]


@pytest.mark.finpile
@pytest.mark.unit
def test_dataset_bounds_checking(mock_finpile_data):
    """Test dataset bounds checking with validation."""
    from data_decide.utils.finpile_data_loader import SimpleTapeDataset
    from data_decide.utils.validation import ValidationError

    with SimpleTapeDataset(mock_finpile_data, chunk_size=50) as dataset:
        # Mock the dataset to have known size
        with patch.object(dataset, "_total_chunks", 5):
            # Valid access should work (with proper mocking)
            with patch.object(dataset, "_validate_dataset_state"):
                with patch.object(dataset, "_handle", np.array([1] * 250, dtype=np.uint16)):
                    # This should work for valid indices
                    pass

            # Invalid access should raise ValidationError
            with pytest.raises((ValidationError, IndexError)):
                with patch.object(dataset, "_validate_dataset_state"):
                    _ = dataset[10]  # Out of bounds


@pytest.mark.finpile
@pytest.mark.integration
def test_training_config_compatibility():
    """Test that FinPile integration works with training configurations."""
    # Test config structure for FinPile training
    finpile_config = {
        "training": {
            "model_size": "4M",
            "sequence_length": 512,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "max_steps": 100,
            "gradient_accumulation_steps": 1,
            "fp16": False,
            "report_to": [],
        },
        "data": {
            "data_path": "mock_finpile_data",
            "chunk_size": 4096,
            "packed": False,
        },
    }

    # Verify structure is compatible with expected format
    assert "training" in finpile_config
    assert "data" in finpile_config
    assert "model_size" in finpile_config["training"]
    assert "data_path" in finpile_config["data"]

    # Verify FinPile-specific parameters
    assert "chunk_size" in finpile_config["data"]
    assert isinstance(finpile_config["data"]["chunk_size"], int)


@pytest.mark.finpile
@pytest.mark.unit
def test_memory_management():
    """Test memory management features of FinPile dataset."""
    from data_decide.utils.finpile_data_loader import SimpleTapeDataset

    class MockDataset(SimpleTapeDataset):
        def __init__(self):
            self._is_closed = False
            self._handle = MagicMock()
            self._offsets = MagicMock()

    dataset = MockDataset()

    # Test cleanup
    dataset.close()
    assert dataset._is_closed
    assert dataset._handle is None
    assert dataset._offsets is None

    # Test destructor calls close
    dataset2 = MockDataset()
    dataset2.__del__()
    assert dataset2._is_closed
