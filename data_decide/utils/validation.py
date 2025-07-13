"""
Comprehensive input validation system with early error detection.

This module provides validation decorators, functions, and classes that catch
errors early in the process with clear, actionable error messages.
"""

from __future__ import annotations

import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .logging_utils import get_logger
from .type_definitions import ExperimentConfig, ModelConfig, TrainingConfig

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Custom Validation Exceptions
# =============================================================================


class ValidationError(ValueError):
    """Base class for validation errors with detailed context."""

    def __init__(
        self, message: str, field: Optional[str] = None, value: Any = None, suggestions: Optional[List[str]] = None
    ):
        self.field = field
        self.value = value
        self.suggestions = suggestions or []

        # Build comprehensive error message
        full_message = message
        if field:
            full_message = f"Validation failed for '{field}': {message}"
        if value is not None:
            full_message += f" (got: {repr(value)})"
        if suggestions:
            full_message += "\n\nSuggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)

        super().__init__(full_message)


class ConfigurationError(ValidationError):
    """Configuration-specific validation error."""

    pass


class DataValidationError(ValidationError):
    """Data-specific validation error."""

    pass


class ModelValidationError(ValidationError):
    """Model-specific validation error."""

    pass


# =============================================================================
# Validation Decorators
# =============================================================================


def validate_config(required_keys: List[str], optional_keys: Optional[List[str]] = None) -> Callable[[F], F]:
    """
    Decorator to validate configuration dictionaries.

    Args:
        required_keys: Keys that must be present
        optional_keys: Keys that may be present (for strict validation)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find the config parameter
            config = None
            if args and isinstance(args[0], dict):
                config = args[0]
            elif "config" in kwargs:
                config = kwargs["config"]
            else:
                # Look for config in first few arguments
                for arg in args[:3]:
                    if isinstance(arg, dict) and any(key in arg for key in required_keys):
                        config = arg
                        break

            if config is not None:
                validate_config_dict(config, required_keys, optional_keys)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_file_path(func: F) -> F:
    """Decorator to validate file path arguments."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for common file path parameter names
        for key in ["file_path", "path", "filepath", "filename"]:
            if key in kwargs:
                validate_file_exists(kwargs[key], key)

        # Check positional arguments that look like file paths
        for i, arg in enumerate(args):
            if isinstance(arg, (str, Path)) and ("/" in str(arg) or "\\" in str(arg)):
                validate_file_exists(arg, f"argument_{i}")

        return func(*args, **kwargs)

    return wrapper


def validate_positive_number(func: F) -> F:
    """Decorator to validate that numeric arguments are positive."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        numeric_params = ["batch_size", "learning_rate", "max_steps", "num_layers", "hidden_size", "num_heads"]

        for key in numeric_params:
            if key in kwargs:
                value = kwargs[key]
                if isinstance(value, (int, float)) and value <= 0:
                    raise ValidationError(
                        "Must be positive",
                        field=key,
                        value=value,
                        suggestions=[f"Use a positive number, e.g., {key}=1 or {key}=0.001"],
                    )

        return func(*args, **kwargs)

    return wrapper


# =============================================================================
# Configuration Validation Functions
# =============================================================================


def validate_config_dict(
    config: Dict[str, Any], required_keys: List[str], optional_keys: Optional[List[str]] = None
) -> None:
    """
    Validate configuration dictionary with detailed error messages.

    Args:
        config: Configuration to validate
        required_keys: Keys that must be present
        optional_keys: Keys that may be present (None for any keys allowed)
    """
    # Check required keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        suggestions = [f"Add missing key: {key}" for key in missing_keys]
        if len(missing_keys) == 1:
            suggestions.append(f"Example: {missing_keys[0]} = <appropriate_value>")

        raise ConfigurationError(f"Missing required configuration keys: {missing_keys}", suggestions=suggestions)

    # Check for unknown keys if strict validation is enabled
    if optional_keys is not None:
        allowed_keys = set(required_keys + optional_keys)
        unknown_keys = [key for key in config.keys() if key not in allowed_keys]
        if unknown_keys:
            suggestions = [f"Remove unknown key: {key}" for key in unknown_keys]
            suggestions.append(f"Allowed keys: {sorted(allowed_keys)}")

            raise ConfigurationError(f"Unknown configuration keys: {unknown_keys}", suggestions=suggestions)


def validate_training_config(config: TrainingConfig) -> None:
    """Validate training configuration with specific checks."""

    # Validate model size
    valid_model_sizes = [
        "4M",
        "6M",
        "8M",
        "10M",
        "14M",
        "16M",
        "20M",
        "60M",
        "90M",
        "150M",
        "300M",
        "530M",
        "750M",
        "1B",
    ]
    if config.get("model_size") not in valid_model_sizes:
        raise ConfigurationError(
            "Invalid model size",
            field="model_size",
            value=config.get("model_size"),
            suggestions=[f"Use one of: {valid_model_sizes}", "Most common sizes: 4M, 150M, 1B"],
        )

    # Validate batch size
    batch_size = config.get("batch_size", 0)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ConfigurationError(
            "Batch size must be a positive integer",
            field="batch_size",
            value=batch_size,
            suggestions=[
                "For development: batch_size=2 or batch_size=4",
                "For training: batch_size=8, batch_size=16, or batch_size=32",
            ],
        )

    # Validate learning rate
    learning_rate = config.get("learning_rate", 0)
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ConfigurationError(
            "Learning rate must be a positive number",
            field="learning_rate",
            value=learning_rate,
            suggestions=[
                "Typical values: 1e-4, 2e-4, 5e-4",
                "For large models: 1e-5 to 1e-4",
                "For small models: 1e-4 to 1e-3",
            ],
        )

    # Validate max_steps
    max_steps = config.get("max_steps", 0)
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ConfigurationError(
            "Max steps must be a positive integer",
            field="max_steps",
            value=max_steps,
            suggestions=[
                "For testing: max_steps=100 or max_steps=1000",
                "For training: max_steps=10000 or max_steps=100000",
                "For full training: max_steps=500000+",
            ],
        )

    # Validate scheduler type
    valid_schedulers = ["linear", "cosine"]
    scheduler_type = config.get("lr_scheduler_type", "")
    if scheduler_type not in valid_schedulers:
        raise ConfigurationError(
            "Invalid learning rate scheduler type",
            field="lr_scheduler_type",
            value=scheduler_type,
            suggestions=[
                f"Use one of: {valid_schedulers}",
                "Most common: 'linear' for fine-tuning, 'cosine' for pre-training",
            ],
        )

    # Validate gradient accumulation
    grad_accum = config.get("gradient_accumulation_steps", 0)
    if not isinstance(grad_accum, int) or grad_accum <= 0:
        raise ConfigurationError(
            "Gradient accumulation steps must be a positive integer",
            field="gradient_accumulation_steps",
            value=grad_accum,
            suggestions=["Typical values: 1, 2, 4, 8", "Use higher values for effective larger batch sizes"],
        )


def validate_model_config(config: ModelConfig) -> None:
    """Validate model configuration with architectural constraints."""

    # Validate vocab size
    vocab_size = config.get("vocab_size", 0)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ModelValidationError(
            "Vocabulary size must be a positive integer",
            field="vocab_size",
            value=vocab_size,
            suggestions=["For FinPile tokenizer: vocab_size=50277", "Common sizes: 32000, 50257, 50277"],
        )

    # Validate hidden size
    hidden_size = config.get("hidden_size", 0)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ModelValidationError(
            "Hidden size must be a positive integer",
            field="hidden_size",
            value=hidden_size,
            suggestions=[
                "Common sizes: 64, 128, 256, 512, 768, 1024",
                "Must be divisible by number of attention heads",
            ],
        )

    # Validate attention heads
    num_heads = config.get("num_attention_heads", 0)
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ModelValidationError(
            "Number of attention heads must be a positive integer",
            field="num_attention_heads",
            value=num_heads,
            suggestions=["Common values: 8, 12, 16, 24, 32", "Must divide hidden_size evenly"],
        )

    # Validate head dimension
    if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
        head_dim = hidden_size // num_heads
        if hidden_size % num_heads != 0:
            raise ModelValidationError(
                f"Hidden size ({hidden_size}) must be divisible by number of heads ({num_heads})",
                suggestions=[
                    f"Adjust hidden_size to be divisible by {num_heads}",
                    f"Or change num_attention_heads to divide {hidden_size}",
                    f"Current head dimension would be: {head_dim} (fractional)",
                ],
            )

        if head_dim < 8 or head_dim > 128:
            raise ModelValidationError(
                f"Head dimension ({head_dim}) outside recommended range [8, 128]",
                suggestions=[
                    "Increase hidden_size or decrease num_attention_heads",
                    "Common head dimensions: 32, 64, 80, 96",
                ],
            )


# =============================================================================
# File and Path Validation
# =============================================================================


def validate_file_exists(path: Union[str, Path], field_name: str = "file_path") -> Path:
    """
    Validate that a file exists and is readable.

    Args:
        path: File path to validate
        field_name: Name of the field for error messages

    Returns:
        Validated Path object
    """
    path_obj = Path(path)

    if not path_obj.exists():
        # Provide helpful suggestions for common mistakes
        suggestions = [
            "Check the file path for typos",
            f"Verify the file exists: ls -la {path_obj.parent}"
            if path_obj.parent.exists()
            else f"Create the directory: mkdir -p {path_obj.parent}",
        ]

        # Check for common alternative locations
        alternatives = []
        if path_obj.name:
            # Check current directory
            current_dir_file = Path.cwd() / path_obj.name
            if current_dir_file.exists():
                alternatives.append(f"Found in current directory: {current_dir_file}")

            # Check common data directories
            for data_dir in ["data", "../data", "datasets", "../datasets"]:
                alt_path = Path(data_dir) / path_obj.name
                if alt_path.exists():
                    alternatives.append(f"Found in {data_dir}: {alt_path}")

        if alternatives:
            suggestions.extend(alternatives)

        raise DataValidationError("File does not exist", field=field_name, value=str(path), suggestions=suggestions)

    if not path_obj.is_file():
        raise DataValidationError(
            "Path is not a file",
            field=field_name,
            value=str(path),
            suggestions=[
                "Ensure the path points to a file, not a directory",
                f"List directory contents: ls -la {path_obj}",
            ],
        )

    if not os.access(path_obj, os.R_OK):
        raise DataValidationError(
            "File is not readable",
            field=field_name,
            value=str(path),
            suggestions=[f"Check file permissions: ls -la {path_obj}", f"Make file readable: chmod +r {path_obj}"],
        )

    return path_obj


def validate_directory_exists(path: Union[str, Path], field_name: str = "directory") -> Path:
    """Validate that a directory exists and is accessible."""
    path_obj = Path(path)

    if not path_obj.exists():
        suggestions = [f"Create the directory: mkdir -p {path_obj}", "Check the path for typos"]

        raise DataValidationError(
            "Directory does not exist", field=field_name, value=str(path), suggestions=suggestions
        )

    if not path_obj.is_dir():
        raise DataValidationError(
            "Path is not a directory",
            field=field_name,
            value=str(path),
            suggestions=["Ensure the path points to a directory, not a file", f"Check: ls -la {path_obj.parent}"],
        )

    if not os.access(path_obj, os.R_OK | os.X_OK):
        raise DataValidationError(
            "Directory is not accessible",
            field=field_name,
            value=str(path),
            suggestions=[
                f"Check directory permissions: ls -la {path_obj}",
                f"Make directory accessible: chmod +rx {path_obj}",
            ],
        )

    return path_obj


# =============================================================================
# Data Validation Functions
# =============================================================================


def validate_dataset_bounds(index: int, dataset_size: int, dataset_name: str = "dataset") -> None:
    """Validate dataset index bounds with clear error messages."""
    if not isinstance(index, int):
        raise DataValidationError(
            "Index must be an integer",
            field="index",
            value=index,
            suggestions=[f"Convert to integer: int({index})", "Ensure you're not using float indices"],
        )

    if index < 0:
        raise DataValidationError(
            f"Negative index {index} is not allowed for {dataset_name} bounds validation",
            field="index",
            value=index,
            suggestions=[
                f"Use positive index: 0 to {dataset_size - 1}",
                f"For last element, use: {dataset_size - 1}",
            ],
        )
    elif index >= dataset_size:
        raise DataValidationError(
            f"Index {index} is out of bounds for {dataset_name} of size {dataset_size}",
            field="index",
            value=index,
            suggestions=[
                f"Use valid index range: 0 to {dataset_size - 1}",
                f"Check dataset size: len({dataset_name}) = {dataset_size}",
                "Ensure dataset is properly loaded",
            ],
        )


def validate_tensor_shape(
    tensor: Any, expected_shape: Optional[List[Optional[int]]] = None, field_name: str = "tensor"
) -> None:
    """Validate tensor shape with detailed error information."""
    if not hasattr(tensor, "shape"):
        raise DataValidationError(
            "Expected tensor-like object with shape attribute",
            field=field_name,
            value=type(tensor).__name__,
            suggestions=["Ensure input is a PyTorch tensor or NumPy array", "Convert to tensor: torch.tensor(data)"],
        )

    if expected_shape is not None:
        actual_shape = list(tensor.shape)

        # Check number of dimensions
        if len(actual_shape) != len(expected_shape):
            raise DataValidationError(
                f"Expected {len(expected_shape)} dimensions, got {len(actual_shape)}",
                field=field_name,
                value=actual_shape,
                suggestions=[
                    f"Expected shape: {expected_shape}",
                    "Add/remove dimensions with reshape, squeeze, or unsqueeze",
                    "Check data preprocessing pipeline",
                ],
            )

        # Check individual dimensions
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected is not None and actual != expected:
                raise DataValidationError(
                    f"Dimension {i}: expected {expected}, got {actual}",
                    field=field_name,
                    value=actual_shape,
                    suggestions=[
                        f"Expected shape: {expected_shape}",
                        f"Reshape tensor: tensor.reshape({expected_shape})",
                        "Check data loading and preprocessing",
                    ],
                )


# =============================================================================
# Early Validation Hooks
# =============================================================================


class EarlyValidator:
    """Context manager for early validation of operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.validations: List[Callable[[], None]] = []
        self.in_context = False

    def add_validation(self, validation_func: Callable[[], None]) -> "EarlyValidator":
        """Add a validation function to be executed."""
        self.validations.append(validation_func)

        # If we're already in context, execute the validation immediately
        if self.in_context:
            try:
                validation_func()
                logger.debug(f"Validation passed for {self.operation_name}")
            except ValidationError as e:
                logger.error(f"Validation failed for {self.operation_name}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in validation for {self.operation_name}: {e}")
                raise ValidationError(
                    f"Validation failed with unexpected error: {e}",
                    suggestions=["Check the validation function implementation"],
                )

        return self

    def __enter__(self) -> "EarlyValidator":
        """Execute all validations before proceeding."""
        self.in_context = True
        logger.debug(f"Running early validations for {self.operation_name}")

        for i, validation in enumerate(self.validations):
            try:
                validation()
            except ValidationError as e:
                logger.error(f"Validation {i + 1} failed for {self.operation_name}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in validation {i + 1} for {self.operation_name}: {e}")
                raise ValidationError(
                    f"Validation failed with unexpected error: {e}",
                    suggestions=["Check the validation function implementation"],
                )

        logger.debug(f"All {len(self.validations)} validations passed for {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up after validation."""
        self.in_context = False


# =============================================================================
# Validation Summary and Reporting
# =============================================================================


def validate_complete_experiment_config(config: Dict[str, Any]) -> ExperimentConfig:
    """
    Comprehensive validation of complete experiment configuration.

    This function performs early validation of all configuration aspects
    and provides detailed, actionable error messages.
    """
    logger.info("Performing comprehensive experiment configuration validation...")

    with EarlyValidator("experiment_config") as validator:
        # Validate top-level structure
        validator.add_validation(
            lambda: validate_config_dict(
                config,
                required_keys=["training", "model", "data"],
                optional_keys=["experiment_name", "output_dir", "wandb_project", "wandb_entity"],
            )
        )

        # Validate training configuration
        if "training" in config:
            validator.add_validation(lambda: validate_training_config(config["training"]))

        # Validate model configuration
        if "model" in config:
            validator.add_validation(lambda: validate_model_config(config["model"]))

        # Validate data paths
        if "data" in config and "data_path" in config["data"]:
            data_path = config["data"]["data_path"]
            if os.path.exists(data_path):
                validator.add_validation(lambda: validate_directory_exists(data_path, "data_path"))

    # Import and use the existing validation from type_definitions
    from .type_definitions import validate_experiment_config

    return validate_experiment_config(config)


def print_validation_summary(config: Dict[str, Any]) -> None:
    """Print a summary of configuration validation results."""
    print("🔍 Configuration Validation Summary")
    print("=" * 40)

    try:
        validated_config = validate_complete_experiment_config(config)
        print("✅ Configuration validation passed")
        print(f"📋 Experiment: {validated_config.get('experiment_name', 'unnamed')}")
        print(f"🏗️  Model size: {validated_config['training']['model_size']}")
        print(f"📊 Batch size: {validated_config['training']['batch_size']}")
        print(f"🎯 Max steps: {validated_config['training']['max_steps']}")
        print(f"📁 Output dir: {validated_config.get('output_dir', './outputs')}")
        return True

    except ValidationError as e:
        print("❌ Configuration validation failed")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print("❌ Unexpected validation error")
        print(f"Error: {e}")
        return False
