"""
Test validation and error handling systems.

This module tests the validation infrastructure (Issue #7) and
consistent error messaging (Issue #8) implementations.
"""

import pytest


@pytest.mark.unit
def test_validation_module_imports():
    """Test that validation module imports work correctly."""
    from data_decide.utils.validation import (
        EarlyValidator,
        ValidationError,
        validate_dataset_bounds,
    )

    # Verify classes/functions are importable
    assert ValidationError is not None
    assert EarlyValidator is not None
    assert validate_dataset_bounds is not None


@pytest.mark.unit
def test_validation_error_creation():
    """Test ValidationError creation and message formatting."""
    from data_decide.utils.validation import ValidationError

    # Test basic error creation
    error = ValidationError("Test validation error", suggestions=["Fix the issue"])
    assert "Test validation error" in str(error)
    assert len(str(error)) > 50  # Should have rich context

    # Test error with field and value
    error_with_context = ValidationError("Invalid value", field="test_field", value="bad_value")
    assert "Invalid value" in str(error_with_context)


@pytest.mark.unit
def test_early_validator_context_manager():
    """Test EarlyValidator context manager functionality."""
    from data_decide.utils.validation import EarlyValidator, ValidationError

    def failing_validation():
        raise ValidationError("Sample validation", suggestions=["Fix it"])

    # Test that validation errors are properly raised
    with pytest.raises(ValidationError) as exc_info:
        with EarlyValidator("test_operation") as validator:
            validator.add_validation(failing_validation)

    assert "Sample validation" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "index,size,should_fail",
    [
        (-1, 100, True),  # Negative index
        (150, 100, True),  # Out of bounds
        (50, 100, False),  # Valid index
        (0, 100, False),  # Edge case: first index
        (99, 100, False),  # Edge case: last index
    ],
)
def test_dataset_bounds_validation(index, size, should_fail):
    """Test dataset bounds validation with various inputs."""
    from data_decide.utils.validation import ValidationError, validate_dataset_bounds

    if should_fail:
        with pytest.raises((ValidationError, ValueError)):
            validate_dataset_bounds(index, size, "test_dataset")
    else:
        # Should not raise any exception
        validate_dataset_bounds(index, size, "test_dataset")


@pytest.mark.unit
def test_error_handling_module_imports():
    """Test that error handling module imports work correctly."""
    from data_decide.utils.error_handling import (
        ErrorCategory,
        ErrorReporter,
        ErrorSeverity,
        build_config_error,
        build_data_loading_error,
    )

    # Verify all components are importable
    assert ErrorReporter is not None
    assert ErrorSeverity is not None
    assert ErrorCategory is not None
    assert build_config_error is not None
    assert build_data_loading_error is not None


@pytest.mark.unit
def test_error_reporter_functionality(error_reporter):
    """Test ErrorReporter basic functionality."""
    from data_decide.utils.error_handling import build_config_error

    # Test error creation and reporting
    config_error = build_config_error(missing_key="model_size", section="training")

    # Verify error has suggestions
    assert len(config_error.context.suggestions) > 0

    # Test error reporting
    error_reporter.report_error(config_error, log_level="warning")
    summary = error_reporter.get_error_summary()

    # Verify error was tracked
    assert "ConfigurationError" in summary
    assert summary["ConfigurationError"] == 1


@pytest.mark.unit
def test_error_message_builders():
    """Test error message builder functions."""
    from data_decide.utils.error_handling import (
        build_config_error,
        build_data_loading_error,
    )

    # Test configuration error
    config_error = build_config_error(missing_key="model_size", section="training")
    assert "model_size" in config_error.context.message
    assert "training" in config_error.context.message
    assert len(config_error.context.suggestions) >= 3

    # Test data loading error
    data_error = build_data_loading_error(file_path="/nonexistent/file.json", error_type="not_found")
    assert "/nonexistent/file.json" in data_error.context.message
    assert len(data_error.context.suggestions) >= 3


@pytest.mark.unit
def test_error_severity_and_category_enums():
    """Test error severity and category enums."""
    from data_decide.utils.error_handling import ErrorCategory, ErrorSeverity

    # Test severity enum values
    assert ErrorSeverity.HIGH.value == "HIGH"
    assert ErrorSeverity.CRITICAL.value == "CRITICAL"
    assert ErrorSeverity.LOW.value == "LOW"

    # Test category enum values
    assert ErrorCategory.CONFIGURATION.value == "configuration"
    assert ErrorCategory.DATA_LOADING.value == "data_loading"
    assert ErrorCategory.VALIDATION.value == "validation"


@pytest.mark.integration
def test_configuration_constants_exist(project_root):
    """Test that configuration constants are properly defined."""
    config_file = project_root / "data_decide" / "olmo" / "models" / "configuration_olmo.py"

    assert config_file.exists(), "Configuration file should exist"

    with open(config_file, "r") as f:
        content = f.read()

    # Check that required constants are defined
    required_constants = [
        "FINPILE_VOCAB_SIZE = 50277",
        "INTERMEDIATE_SIZE_RATIO = 4",
        "HEAD_DIM = 64",
        "MODEL_SCALING_CONFIG = {",
    ]

    for constant in required_constants:
        assert constant in content, f"Missing constant: {constant}"

    # Check that factory class exists
    required_factory_methods = [
        "class ModelConfigFactory:",
        "def create_config(",
        "def get_available_sizes(",
        "def estimate_parameters(",
    ]

    for method in required_factory_methods:
        assert method in content, f"Missing factory method: {method}"


@pytest.mark.integration
def test_data_loader_safety_features(project_root):
    """Test that data loader has required safety features."""
    loader_file = project_root / "data_decide" / "utils" / "finpile_data_loader.py"

    assert loader_file.exists(), "Data loader file should exist"

    with open(loader_file, "r") as f:
        content = f.read()

    # Check for context manager methods
    context_manager_features = [
        "def __enter__(self):",
        "def __exit__(self, exc_type, exc_val, exc_tb):",
        "def close(self):",
        "self._is_closed",
    ]

    for feature in context_manager_features:
        assert feature in content, f"Missing safety feature: {feature}"

    # Check for safety validations
    safety_checks = [
        "if self._is_closed:",
        "Cannot access data from a closed dataset",
    ]

    for check in safety_checks:
        assert check in content, f"Missing safety check: {check}"


@pytest.mark.integration
def test_error_handling_patterns_in_data_curation(project_root):
    """Test that data curation has comprehensive error handling."""
    curation_file = project_root / "data_decide" / "olmo" / "data" / "data_curation.py"

    assert curation_file.exists(), "Data curation file should exist"

    with open(curation_file, "r") as f:
        content = f.read()

    # Check for comprehensive error handling
    error_handling_patterns = [
        "try:",
        "except (OSError, IOError, UnicodeDecodeError)",
        "except json.JSONDecodeError",
        "logger.warning",
        "logger.error",
    ]

    for pattern in error_handling_patterns:
        assert pattern in content, f"Missing error handling pattern: {pattern}"

    # Check for input validation
    validation_patterns = [
        "if not line:",  # Empty line handling
        "if not data_files:",  # No files handling
        "failed_files",  # Failed file tracking
    ]

    for pattern in validation_patterns:
        assert pattern in content, f"Missing validation pattern: {pattern}"


@pytest.mark.integration
def test_trainer_type_annotations(project_root):
    """Test that trainer has proper type annotations."""
    trainer_file = project_root / "data_decide" / "olmo" / "training" / "trainer.py"

    assert trainer_file.exists(), "Trainer file should exist"

    with open(trainer_file, "r") as f:
        content = f.read()

    # Check for type annotation imports
    type_imports = [
        "from __future__ import annotations",
        "from typing import",
        "from ...utils.type_definitions import",
    ]

    for import_stmt in type_imports:
        assert import_stmt in content, f"Missing type import: {import_stmt}"

    # Check for specific type annotations
    type_annotations = [
        "-> None:",  # Return type annotations
        ": ConfigDict",  # Variable type annotations
        ": TrainingConfig",
        ": MetricsDict",
        "Union[",  # Union types
        "Optional[",  # Optional types
    ]

    for annotation in type_annotations:
        assert annotation in content, f"Missing type annotation: {annotation}"

    # Check for protocol usage
    protocols = [
        "TokenizerProtocol",
        "DatasetProtocol",
    ]

    for protocol in protocols:
        assert protocol in content, f"Missing protocol: {protocol}"


@pytest.mark.integration
def test_deprecated_logging_wrapper(project_root):
    """Test that deprecated logging wrapper is properly implemented."""
    deprecated_file = project_root / "data_decide" / "olmo" / "utils" / "logging_utils.py"

    assert deprecated_file.exists(), "Deprecated logging file should exist"

    with open(deprecated_file, "r") as f:
        content = f.read()

    # Check for deprecation warnings
    deprecation_features = [
        "warnings.warn",
        "deprecated",
        "DeprecationWarning",
    ]

    for feature in deprecation_features:
        assert feature.lower() in content.lower(), f"Missing deprecation feature: {feature}"

    # Check for forwarding imports
    assert "from ...utils.logging_utils import" in content

    # Check for wrapper functions
    wrapper_functions = [
        "def setup_logging(",
        "def get_logger(",
    ]

    for func in wrapper_functions:
        assert func in content, f"Missing wrapper function: {func}"


@pytest.mark.integration
def test_core_file_structure(project_root):
    """Test that all expected core files exist."""
    expected_files = [
        "data_decide/utils/type_definitions.py",
        "data_decide/utils/logging_utils.py",
        "data_decide/olmo/utils/logging_utils.py",
        "data_decide/olmo/models/configuration_olmo.py",
        "data_decide/olmo/training/trainer.py",
        "data_decide/olmo/data/data_curation.py",
        "data_decide/utils/finpile_data_loader.py",
    ]

    for file_path in expected_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"Expected file missing: {file_path}"

        # Check that file is not empty
        assert full_path.stat().st_size > 0, f"File is empty: {file_path}"
