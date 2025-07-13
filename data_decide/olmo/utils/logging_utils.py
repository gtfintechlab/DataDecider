"""OLMo training logging utilities - DEPRECATED.

This module is deprecated. Use data_decide.utils.logging_utils for all logging functionality.
"""

import logging
import warnings
from typing import Optional

from ...utils.logging_utils import get_logger as _get_logger

# Import from the consolidated logging utilities
from ...utils.logging_utils import setup_logging as _setup_logging

# Deprecation warning
warnings.warn(
    "data_decide.olmo.utils.logging_utils is deprecated. Use data_decide.utils.logging_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)


def setup_logging(output_dir: str, log_level: str = "INFO", log_file: str = "training.log") -> None:
    """
    DEPRECATED: Setup logging configuration for training.

    Use data_decide.utils.logging_utils.configure_for_training() instead.

    Args:
        output_dir: Directory to save log files
        log_level: Logging level (INFO, DEBUG, etc.)
        log_file: Name of the log file
    """
    warnings.warn(
        "setup_logging is deprecated. Use configure_for_training() from data_decide.utils.logging_utils",
        DeprecationWarning,
        stacklevel=2,
    )

    # Forward to the new consolidated function
    _setup_logging(output_dir=output_dir, log_level=log_level, log_file=log_file)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    DEPRECATED: Get a logger instance.

    Use data_decide.utils.logging_utils.get_logger() instead.

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        Logger instance
    """
    warnings.warn(
        "get_logger is deprecated. Use get_logger() from data_decide.utils.logging_utils",
        DeprecationWarning,
        stacklevel=2,
    )

    return _get_logger(name)
