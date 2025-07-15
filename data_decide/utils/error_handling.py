"""
Consistent error messaging and handling patterns.

This module provides standardized error messages, error recovery strategies,
and consistent error reporting across the entire codebase.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Error Categories and Severity Levels
# =============================================================================


class ErrorSeverity(Enum):
    """Standardized error severity levels."""

    CRITICAL = "CRITICAL"  # System cannot continue, immediate action required
    HIGH = "HIGH"  # Major functionality broken, urgent fix needed
    MEDIUM = "MEDIUM"  # Important feature affected, should fix soon
    LOW = "LOW"  # Minor issue, can be fixed later
    WARNING = "WARNING"  # Potential issue, informational


class ErrorCategory(Enum):
    """Standardized error categories for consistent messaging."""

    CONFIGURATION = "configuration"
    DATA_LOADING = "data_loading"
    MODEL_TRAINING = "model_training"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    MEMORY = "memory"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    DEPENDENCY = "dependency"


# =============================================================================
# Standardized Error Messages
# =============================================================================


class ErrorMessages:
    """Centralized error message templates with consistent formatting."""

    # Configuration errors
    MISSING_CONFIG_KEY = "Missing required configuration key '{key}'. Expected in section '{section}'."
    INVALID_CONFIG_VALUE = "Invalid value for '{key}': expected {expected}, got {actual}."
    INCOMPATIBLE_CONFIG = "Configuration conflict: {field1}={value1} is incompatible with {field2}={value2}."

    # Data loading errors
    FILE_NOT_FOUND = "Data file not found: '{path}'. Check that the file exists and is readable."
    CORRUPTED_DATA = "Data corruption detected in '{path}' at {location}. File may be incomplete or damaged."
    INVALID_DATA_FORMAT = "Invalid data format in '{path}': expected {expected_format}, found {actual_format}."
    DATASET_EMPTY = "Dataset '{name}' is empty or contains no valid samples."
    INDEX_OUT_OF_BOUNDS = "Index {index} is out of bounds for dataset '{name}' (size: {size})."

    # Model training errors
    TRAINING_CONVERGENCE_FAILURE = "Training failed to converge after {steps} steps. Loss: {loss}."
    GPU_MEMORY_ERROR = "GPU out of memory. Try reducing batch_size from {current} to {suggested}."
    MODEL_LOAD_FAILURE = "Failed to load model from '{path}': {reason}."
    CHECKPOINT_CORRUPTED = "Checkpoint '{path}' is corrupted or incompatible with current model."

    # File system errors
    PERMISSION_DENIED = "Permission denied accessing '{path}'. Check file permissions."
    DISK_SPACE_ERROR = "Insufficient disk space for operation. Need {required}GB, have {available}GB."
    DIRECTORY_NOT_FOUND = "Directory '{path}' does not exist. Create it or check the path."

    # Memory errors
    OUT_OF_MEMORY = "System out of memory during {operation}. Try reducing data size or batch size."
    MEMORY_LEAK_DETECTED = "Potential memory leak detected: {memory_usage}MB allocated."

    # Validation errors
    INVALID_INPUT_TYPE = "Invalid input type for '{parameter}': expected {expected}, got {actual}."
    VALUE_OUT_OF_RANGE = "Value {value} for '{parameter}' is out of range [{min_val}, {max_val}]."
    VALIDATION_FAILED = "Validation failed for '{field}': {reason}."

    # Network/dependency errors
    DEPENDENCY_MISSING = "Required dependency '{name}' not found. Install with: {install_command}"
    CONNECTION_FAILED = "Failed to connect to {service}: {reason}."
    TIMEOUT_ERROR = "Operation timed out after {timeout}s while {operation}."


# =============================================================================
# Error Context and Suggestions
# =============================================================================


class ErrorContext:
    """Rich error context with troubleshooting information."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        suggestions: Optional[List[str]] = None,
        documentation_url: Optional[str] = None,
        error_code: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.suggestions = suggestions or []
        self.documentation_url = documentation_url
        self.error_code = error_code
        self.context_data = context_data or {}

    def add_suggestion(self, suggestion: str) -> "ErrorContext":
        """Add a troubleshooting suggestion."""
        self.suggestions.append(suggestion)
        return self

    def add_context(self, key: str, value: Any) -> "ErrorContext":
        """Add context information."""
        self.context_data[key] = value
        return self

    def format_full_message(self) -> str:
        """Format the complete error message with all context."""
        lines = [
            f"🚨 {self.severity.value} ERROR ({self.category.value})",
            "",
            f"Message: {self.message}",
        ]

        if self.error_code:
            lines.append(f"Error Code: {self.error_code}")

        if self.context_data:
            lines.append("")
            lines.append("Context:")
            for key, value in self.context_data.items():
                lines.append(f"  {key}: {value}")

        if self.suggestions:
            lines.append("")
            lines.append("Troubleshooting Suggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")

        if self.documentation_url:
            lines.append("")
            lines.append(f"Documentation: {self.documentation_url}")

        return "\n".join(lines)


# =============================================================================
# Enhanced Exception Classes
# =============================================================================


class DataDeciderError(Exception):
    """Base exception for all DataDecider-specific errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.VALIDATION,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        suggestions: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.context = ErrorContext(
            message=message, category=category, severity=severity, suggestions=suggestions, error_code=error_code
        )
        self.original_exception = original_exception

        super().__init__(self.context.format_full_message())


class ConfigurationError(DataDeciderError):
    """Configuration-related errors with specific troubleshooting."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault(
            "suggestions",
            [
                "Check configuration file syntax and structure",
                "Validate all required fields are present",
                "Compare with working configuration examples",
            ],
        )
        super().__init__(message, **kwargs)


class DataLoadingError(DataDeciderError):
    """Data loading and processing errors."""

    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        kwargs.setdefault("category", ErrorCategory.DATA_LOADING)
        kwargs.setdefault(
            "suggestions",
            [
                "Verify file exists and is readable",
                "Check file format and structure",
                "Ensure sufficient permissions",
                "Try with a smaller sample first",
            ],
        )

        if file_path:
            kwargs.setdefault("context_data", {})["file_path"] = file_path

        super().__init__(message, **kwargs)


class TrainingError(DataDeciderError):
    """Model training and optimization errors."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.MODEL_TRAINING)
        kwargs.setdefault(
            "suggestions",
            [
                "Check model configuration parameters",
                "Verify training data is properly formatted",
                "Monitor GPU memory usage",
                "Try reducing batch size or model size",
            ],
        )
        super().__init__(message, **kwargs)


class ValidationError(DataDeciderError):
    """Input validation and constraint errors."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, **kwargs):
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault(
            "suggestions",
            [
                "Check input parameter types and ranges",
                "Validate against expected schema",
                "Refer to documentation for valid values",
            ],
        )

        context_data = kwargs.setdefault("context_data", {})
        if field:
            context_data["field"] = field
        if value is not None:
            context_data["value"] = repr(value)

        super().__init__(message, **kwargs)


# =============================================================================
# Error Recovery and Reporting
# =============================================================================


class ErrorRecovery:
    """Automated error recovery strategies."""

    @staticmethod
    def suggest_batch_size_reduction(current_batch_size: int, memory_error: bool = True) -> List[str]:
        """Suggest batch size reductions for memory errors."""
        suggestions = []

        if memory_error:
            # Aggressive reduction for memory errors
            new_sizes = [current_batch_size // 2, current_batch_size // 4, 1]
        else:
            # Conservative reduction for other issues
            new_sizes = [max(1, current_batch_size - 1), max(1, current_batch_size // 2)]

        for size in new_sizes:
            if size > 0 and size != current_batch_size:
                suggestions.append(f"Try batch_size={size}")

        suggestions.append("Use gradient accumulation to maintain effective batch size")
        return suggestions

    @staticmethod
    def suggest_file_recovery(file_path: str, error_type: str) -> List[str]:
        """Suggest file-related error recovery."""
        path_obj = Path(file_path)
        suggestions = []

        if error_type == "not_found":
            suggestions.extend(
                [
                    f"Check if file exists: ls -la {path_obj}",
                    f"Check parent directory: ls -la {path_obj.parent}",
                    "Verify the file path is correct",
                    "Check for typos in the filename",
                ]
            )

            # Look for similar files
            if path_obj.parent.exists():
                similar_files = [
                    f for f in path_obj.parent.iterdir() if f.is_file() and f.stem.lower() == path_obj.stem.lower()
                ]
                if similar_files:
                    suggestions.append(f"Similar files found: {[str(f) for f in similar_files[:3]]}")

        elif error_type == "permission":
            suggestions.extend(
                [
                    f"Check file permissions: ls -la {path_obj}",
                    f"Make file readable: chmod +r {path_obj}",
                    "Check if you have access to the directory",
                    "Try running with appropriate user permissions",
                ]
            )

        elif error_type == "corrupted":
            suggestions.extend(
                [
                    "Try re-downloading or regenerating the file",
                    "Check file integrity with checksums",
                    "Verify disk space and file system health",
                    "Use backup copy if available",
                ]
            )

        return suggestions

    @staticmethod
    def suggest_memory_optimization(operation: str, current_usage: Optional[str] = None) -> List[str]:
        """Suggest memory optimization strategies."""
        suggestions = [
            "Reduce batch size or model size",
            "Use gradient checkpointing to trade compute for memory",
            "Enable mixed precision training (fp16)",
            "Use data loading with num_workers=0 to reduce memory overhead",
        ]

        if operation == "data_loading":
            suggestions.extend(
                [
                    "Load data in streaming mode instead of loading all at once",
                    "Use memory mapping for large datasets",
                    "Clear unnecessary variables with del and gc.collect()",
                ]
            )
        elif operation == "model_training":
            suggestions.extend(
                [
                    "Use model parallelism for very large models",
                    "Consider using DeepSpeed or similar optimization libraries",
                    "Monitor GPU memory with nvidia-smi",
                ]
            )

        if current_usage:
            suggestions.append(f"Current memory usage: {current_usage}")

        return suggestions


# =============================================================================
# Centralized Error Reporter
# =============================================================================


class ErrorReporter:
    """Centralized error reporting and logging."""

    def __init__(self, enable_debug: bool = False):
        self.enable_debug = enable_debug
        self.error_counts: Dict[str, int] = {}

    def report_error(
        self,
        error: Union[DataDeciderError, Exception],
        context: Optional[Dict[str, Any]] = None,
        log_level: str = "error",
    ) -> None:
        """Report error with appropriate logging and context."""

        # Track error frequency
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Build error report
        if isinstance(error, DataDeciderError):
            message = str(error)
            category = error.context.category.value
            severity = error.context.severity.value
        else:
            message = str(error)
            category = "unknown"
            severity = "medium"

        # Add context information
        if context:
            message += f"\n\nAdditional Context: {context}"

        # Add frequency information if this is a repeated error
        if self.error_counts[error_type] > 1:
            message += f"\n\nNote: This error has occurred {self.error_counts[error_type]} times"

        # Log with appropriate level
        if log_level == "critical":
            logger.critical(f"[{category.upper()}] {message}")
        elif log_level == "error":
            logger.error(f"[{category.upper()}] {message}")
        elif log_level == "warning":
            logger.warning(f"[{category.upper()}] {message}")
        else:
            logger.info(f"[{category.upper()}] {message}")

        # Include stack trace for debugging
        if self.enable_debug:
            logger.debug("Stack trace:", exc_info=True)

    def report_recovery_attempt(self, error_type: str, recovery_action: str, success: bool) -> None:
        """Report error recovery attempts."""
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Recovery attempt for {error_type}: {recovery_action} - {status}")

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error frequencies."""
        return self.error_counts.copy()


# =============================================================================
# Global Error Handler Instance
# =============================================================================

# Global error reporter instance
_error_reporter = ErrorReporter()


def get_error_reporter() -> ErrorReporter:
    """Get the global error reporter instance."""
    return _error_reporter


def report_error(error: Union[DataDeciderError, Exception], **kwargs) -> None:
    """Convenience function to report errors globally."""
    _error_reporter.report_error(error, **kwargs)


# =============================================================================
# Error Message Builders
# =============================================================================


def build_config_error(
    missing_key: Optional[str] = None, invalid_value: Optional[tuple] = None, section: str = "configuration"
) -> ConfigurationError:
    """Build standardized configuration error."""

    if missing_key:
        message = ErrorMessages.MISSING_CONFIG_KEY.format(key=missing_key, section=section)
        suggestions = [
            f"Add the missing key: {missing_key} = <value>",
            f"Check the {section} section in your config file",
            "Refer to configuration documentation for required fields",
        ]
    elif invalid_value:
        key, expected, actual = invalid_value
        message = ErrorMessages.INVALID_CONFIG_VALUE.format(key=key, expected=expected, actual=actual)
        suggestions = [
            f"Change {key} to a valid {expected} value",
            f"Current value '{actual}' is not acceptable",
            "Check configuration documentation for valid values",
        ]
    else:
        message = "Configuration error occurred"
        suggestions = ["Check your configuration file for errors"]

    return ConfigurationError(
        message=message, severity=ErrorSeverity.HIGH, suggestions=suggestions, error_code="CONFIG_001"
    )


def build_data_loading_error(file_path: str, error_type: str, details: Optional[str] = None) -> DataLoadingError:
    """Build standardized data loading error."""

    if error_type == "not_found":
        message = ErrorMessages.FILE_NOT_FOUND.format(path=file_path)
        suggestions = ErrorRecovery.suggest_file_recovery(file_path, "not_found")
        error_code = "DATA_001"
    elif error_type == "corrupted":
        location = details or "unknown location"
        message = ErrorMessages.CORRUPTED_DATA.format(path=file_path, location=location)
        suggestions = ErrorRecovery.suggest_file_recovery(file_path, "corrupted")
        error_code = "DATA_002"
    elif error_type == "permission":
        message = ErrorMessages.PERMISSION_DENIED.format(path=file_path)
        suggestions = ErrorRecovery.suggest_file_recovery(file_path, "permission")
        error_code = "DATA_003"
    else:
        message = f"Data loading error with {file_path}: {details or 'unknown error'}"
        suggestions = ["Check file and try again"]
        error_code = "DATA_999"

    return DataLoadingError(
        message=message,
        file_path=file_path,
        severity=ErrorSeverity.HIGH,
        suggestions=suggestions,
        error_code=error_code,
    )


def build_training_error(error_type: str, details: Optional[Dict[str, Any]] = None) -> TrainingError:
    """Build standardized training error."""
    details = details or {}

    if error_type == "memory":
        current_batch = details.get("batch_size", "unknown")
        message = ErrorMessages.GPU_MEMORY_ERROR.format(
            current=current_batch,
            suggested=max(1, int(current_batch) // 2) if isinstance(current_batch, int) else "smaller",
        )
        suggestions = ErrorRecovery.suggest_batch_size_reduction(
            current_batch if isinstance(current_batch, int) else 8, memory_error=True
        )
        error_code = "TRAIN_001"
    elif error_type == "convergence":
        steps = details.get("steps", "unknown")
        loss = details.get("loss", "unknown")
        message = ErrorMessages.TRAINING_CONVERGENCE_FAILURE.format(steps=steps, loss=loss)
        suggestions = [
            "Try reducing learning rate",
            "Check data quality and preprocessing",
            "Increase number of training steps",
            "Verify model architecture is appropriate",
        ]
        error_code = "TRAIN_002"
    else:
        message = f"Training error: {error_type}"
        suggestions = ["Check training configuration and data"]
        error_code = "TRAIN_999"

    return TrainingError(
        message=message,
        severity=ErrorSeverity.HIGH,
        suggestions=suggestions,
        error_code=error_code,
        context_data=details,
    )
