"""
Shared pytest fixtures for DataDecider tests.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_config():
    """Sample training configuration for testing."""
    return {
        "training": {
            "model_size": "4M",
            "batch_size": 4,
            "learning_rate": 1e-4,
            "max_steps": 100,
            "num_train_epochs": 1,
            "gradient_accumulation_steps": 1,
            "fp16": False,
            "report_to": [],
            "num_workers": 0,
            "per_device_eval_batch_size": 2,
            "logging_steps": 10,
            "eval_steps": 50,
            "save_steps": 100,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "warmup_steps": 10,
            "lr_scheduler_type": "linear",
        },
        "model": {
            "vocab_size": 50277,
            "hidden_size": 64,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
        },
        "data": {
            "data_path": "/tmp",
            "max_seq_length": 512,
            "chunk_size": 4096,
        },
        "experiment_name": "test_experiment",
        "output_dir": "/tmp/test_output",
    }


@pytest.fixture
def invalid_config():
    """Invalid configuration for testing validation."""
    return {
        "training": {
            # Missing required model_size
            "batch_size": 8,
            "learning_rate": 1e-4,
            "max_steps": 1000,
        },
        "model": {
            "vocab_size": 50277,
            "hidden_size": 64,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
        },
        "data": {"data_path": "/tmp"},
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing."""
    batch_size = 2
    seq_length = 10

    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long),
        "labels": torch.randint(0, 1000, (batch_size, seq_length)),
    }


@pytest.fixture
def mock_finpile_data(temp_dir):
    """Create mock FinPile data files for testing."""
    prefix = temp_dir / "mock_finpile"

    # Create mock .bin file (small dataset)
    num_tokens = 1000
    tokens = np.random.randint(0, 50277, size=num_tokens, dtype=np.uint16)
    bin_file = f"{prefix}.bin"
    tokens.tofile(bin_file)

    # Create mock .idx file (document offsets)
    num_docs = 10
    doc_lengths = np.random.randint(50, 150, size=num_docs)
    offsets = np.cumsum([0] + doc_lengths.tolist())[:-1]
    idx_file = f"{prefix}.idx"
    offsets.astype(np.int64).tofile(idx_file)

    return str(prefix)


@pytest.fixture
def sample_jsonl_data():
    """Sample JSONL data for data curation testing."""
    return [
        {"text": "Sample document 1 for testing data curation.", "uuid": "doc1"},
        {"text": "Another sample document with different content.", "uuid": "doc2"},
        {"text": "Third document for diversity testing.", "uuid": "doc3"},
        {"text": "Quality document with good content structure.", "uuid": "doc4"},
        {"text": "Final test document for validation.", "uuid": "doc5"},
    ]


@pytest.fixture
def create_jsonl_file(temp_dir):
    """Factory fixture to create JSONL test files."""

    def _create_file(filename: str, data: List[Dict[str, Any]]):
        file_path = temp_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return file_path

    return _create_file


@pytest.fixture
def error_reporter():
    """Create a test error reporter instance."""
    from data_decide.utils.error_handling import ErrorReporter

    return ErrorReporter(enable_debug=False)


@pytest.fixture
def mock_model_config():
    """Mock model configuration for testing."""
    return {
        "vocab_size": 50277,
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "intermediate_size": 256,
        "max_position_embeddings": 2048,
        "type_vocab_size": 1,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-5,
    }


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Cleanup test artifacts after each test."""
    yield
    # Clean up any test artifacts that might have been created
    test_dirs = ["test_output", "checkpoints", "logs"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)


# Test markers for categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow tests")
    config.addinivalue_line("markers", "finpile: marks tests that require FinPile data")
