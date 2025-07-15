"""
Test comprehensive validation system functionality.

This module tests the complete validation system including configuration
validation, early validation, and validation summary features.
"""

import pytest


@pytest.mark.unit
def test_validation_system_imports():
    """Test that validation system imports work correctly."""
    from data_decide.utils.validation import (
        print_validation_summary,
        validate_complete_experiment_config,
    )

    assert validate_complete_experiment_config is not None
    assert print_validation_summary is not None


@pytest.mark.unit
def test_complete_experiment_config_validation(sample_config):
    """Test complete experiment configuration validation."""
    from data_decide.utils.validation import validate_complete_experiment_config

    # Test with valid configuration
    validated = validate_complete_experiment_config(sample_config)

    # Should return the validated config
    assert validated is not None
    assert isinstance(validated, dict)

    # Should contain all required sections
    assert "training" in validated
    assert "model" in validated
    assert "data" in validated


@pytest.mark.unit
def test_validation_summary(sample_config, capsys):
    """Test validation summary printing."""
    from data_decide.utils.validation import print_validation_summary

    # Test validation summary
    print_validation_summary(sample_config)

    # Capture output
    captured = capsys.readouterr()

    # Should contain relevant information
    assert len(captured.out) > 0
    # The summary should show some configuration details
    assert any(key in captured.out for key in ["model_size", "batch_size", "experiment"])


@pytest.mark.unit
def test_invalid_configuration_validation(invalid_config):
    """Test validation with invalid configurations."""
    from data_decide.utils.validation import (
        ValidationError,
        validate_complete_experiment_config,
    )

    # Should raise ValidationError or ValueError for invalid config
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        validate_complete_experiment_config(invalid_config)

    error_msg = str(exc_info.value)

    # Error message should be informative
    assert len(error_msg) > 50
    # Should mention the missing field or contain suggestions
    assert "model_size" in error_msg or "suggestions" in error_msg.lower()


@pytest.mark.unit
def test_early_validation_integration():
    """Test early validation system integration."""
    from data_decide.utils.validation import EarlyValidator, ValidationError

    def sample_validation():
        raise ValidationError("Sample validation error", suggestions=["Fix the issue"])

    # Test that early validation properly catches and reports errors
    with pytest.raises(ValidationError) as exc_info:
        with EarlyValidator("test_operation") as validator:
            validator.add_validation(sample_validation)

    # Verify error has proper context
    assert "Sample validation error" in str(exc_info.value)
    assert len(str(exc_info.value)) > 50  # Should have rich error context


@pytest.mark.unit
def test_configuration_validation_edge_cases():
    """Test configuration validation with various edge cases."""
    from data_decide.utils.validation import (
        ValidationError,
        validate_complete_experiment_config,
    )

    # Test completely empty config
    with pytest.raises((ValidationError, ValueError, KeyError)):
        validate_complete_experiment_config({})

    # Test config with empty training section
    with pytest.raises((ValidationError, ValueError, KeyError)):
        validate_complete_experiment_config({"training": {}})

    # Test config with missing data section
    incomplete_config = {
        "training": {
            "model_size": "4M",
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
        # Missing data section
    }

    with pytest.raises((ValidationError, ValueError, KeyError)):
        validate_complete_experiment_config(incomplete_config)


@pytest.mark.unit
def test_early_validator_with_multiple_validations():
    """Test EarlyValidator with multiple validation functions."""
    from data_decide.utils.validation import EarlyValidator, ValidationError

    def validation1():
        # This should pass
        pass

    def validation2():
        raise ValidationError("Second validation failed")

    # Should fail on the second validation
    with pytest.raises(ValidationError) as exc_info:
        with EarlyValidator("multi_validation_test") as validator:
            validator.add_validation(validation1)
            validator.add_validation(validation2)

    assert "Second validation failed" in str(exc_info.value)


@pytest.mark.integration
def test_validation_with_real_config_structures():
    """Test validation with realistic configuration structures."""
    from data_decide.utils.validation import validate_complete_experiment_config

    # Test with a more complex realistic config
    complex_config = {
        "training": {
            "model_size": "150M",
            "batch_size": 16,
            "learning_rate": 5e-5,
            "max_steps": 10000,
            "num_train_epochs": 3,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "report_to": ["wandb"],
            "num_workers": 4,
            "per_device_eval_batch_size": 8,
            "logging_steps": 100,
            "eval_steps": 1000,
            "save_steps": 2000,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "warmup_steps": 1000,
            "lr_scheduler_type": "cosine",
        },
        "model": {
            "vocab_size": 50277,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        },
        "data": {
            "data_path": "/path/to/data",
            "max_seq_length": 2048,
            "chunk_size": 4096,
            "validation_split": 0.1,
        },
        "experiment_name": "complex_experiment",
        "output_dir": "/path/to/output",
        "logging": {
            "log_level": "INFO",
            "save_logs": True,
        },
    }

    # Should validate successfully
    validated = validate_complete_experiment_config(complex_config)
    assert validated is not None
    assert "training" in validated
    assert validated["training"]["model_size"] == "150M"


@pytest.mark.integration
def test_validation_error_message_quality():
    """Test that validation error messages are helpful and actionable."""
    from data_decide.utils.validation import (
        ValidationError,
        validate_complete_experiment_config,
    )

    # Test with config that has type errors
    bad_type_config = {
        "training": {
            "model_size": "4M",
            "batch_size": "eight",  # Should be int
            "learning_rate": "small",  # Should be float
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

    with pytest.raises((ValidationError, ValueError, TypeError)) as exc_info:
        validate_complete_experiment_config(bad_type_config)

    error_msg = str(exc_info.value)

    # Error message should be descriptive
    assert len(error_msg) > 30
    # Should ideally mention the problematic field or provide context
    # (exact validation depends on implementation)
