"""Comprehensive logging utilities for DataDecider."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union


class DataDeciderLoggingConfig:
    """Centralized logging configuration for DataDecider."""

    # Standard formats
    CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    FILE_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Logger names
    PACKAGE_LOGGER = "data_decide"

    _configured = False


def setup_logging(
    output_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_file: str = "training.log",
    console_level: Optional[str] = None,
) -> None:
    """
    Setup comprehensive logging configuration for DataDecider.

    Args:
        output_dir: Directory to save log files (None for console only)
        log_level: Default logging level (INFO, DEBUG, etc.)
        log_file: Name of the log file
        console_level: Override console logging level (defaults to log_level)
    """
    if DataDeciderLoggingConfig._configured:
        return  # Avoid double configuration

    # Get root logger for the package
    logger = logging.getLogger(DataDeciderLoggingConfig.PACKAGE_LOGGER)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_level_val = getattr(logging, (console_level or log_level).upper())
    console_handler.setLevel(console_level_val)
    console_formatter = logging.Formatter(DataDeciderLoggingConfig.CONSOLE_FORMAT, DataDeciderLoggingConfig.DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_path = Path(output_dir) / log_file

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(DataDeciderLoggingConfig.FILE_FORMAT, DataDeciderLoggingConfig.DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging initialized. Log file: {log_path}")
    else:
        logger.info("Console logging initialized.")

    DataDeciderLoggingConfig._configured = True


def get_logger(name: Optional[str] = None, level: Optional[Union[int, str]] = None) -> logging.Logger:
    """Get a logger with consistent formatting.

    Args:
        name: Logger name (defaults to module name)
        level: Override logging level for this logger

    Returns:
        Configured logger
    """
    # Ensure package-level logging is set up
    if not DataDeciderLoggingConfig._configured:
        setup_logging()  # Initialize with defaults

    # Create child logger under the package
    if name:
        if not name.startswith(DataDeciderLoggingConfig.PACKAGE_LOGGER):
            logger_name = f"{DataDeciderLoggingConfig.PACKAGE_LOGGER}.{name}"
        else:
            logger_name = name
    else:
        logger_name = DataDeciderLoggingConfig.PACKAGE_LOGGER

    logger = logging.getLogger(logger_name)

    # Set specific level if provided
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)

    return logger


def set_verbosity(verbose: bool = False, quiet: bool = False) -> None:
    """Set global logging verbosity for DataDecider.

    Args:
        verbose: Enable verbose logging (DEBUG level)
        quiet: Suppress most logging (WARNING level)
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Set level for all DataDecider loggers
    logger = logging.getLogger(DataDeciderLoggingConfig.PACKAGE_LOGGER)
    logger.setLevel(level)

    # Also update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)


def configure_for_training(output_dir: str, experiment_name: str = "experiment") -> None:
    """Configure logging specifically for training runs.

    Args:
        output_dir: Directory for training outputs
        experiment_name: Name of the experiment for log file naming
    """
    log_file = f"{experiment_name}_{os.getpid()}.log"
    setup_logging(output_dir=output_dir, log_level="INFO", log_file=log_file, console_level="INFO")


def reset_logging() -> None:
    """Reset logging configuration for testing or re-initialization."""
    logger = logging.getLogger(DataDeciderLoggingConfig.PACKAGE_LOGGER)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    DataDeciderLoggingConfig._configured = False
