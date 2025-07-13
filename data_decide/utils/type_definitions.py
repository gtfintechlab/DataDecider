"""Comprehensive type definitions for DataDecider project.

This module provides TypedDict classes, Protocol interfaces, and type aliases
for improved type safety and better IDE support throughout the codebase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TypedDict

import torch

# =============================================================================
# Type Aliases
# =============================================================================

TensorDict = Dict[str, torch.Tensor]
MetricsDict = Dict[str, float]
ConfigDict = Dict[str, Any]


# =============================================================================
# Configuration TypedDict Classes
# =============================================================================


class TrainingConfig(TypedDict, total=False):
    """Type definition for training configuration."""

    # Model configuration
    model_size: str

    # Training parameters
    batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: int
    max_steps: int
    gradient_accumulation_steps: int

    # Optimization
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float

    # Learning rate scheduling
    lr_scheduler_type: str  # "linear" | "cosine"
    warmup_steps: int

    # Hardware and performance
    fp16: bool
    num_workers: int

    # Logging and evaluation
    logging_steps: int
    eval_steps: int
    save_steps: int
    report_to: List[str]  # ["wandb", "tensorboard", etc.]

    # Optional advanced settings
    dataloader_drop_last: Optional[bool]
    dataloader_num_workers: Optional[int]
    disable_tqdm: Optional[bool]


class ModelConfig(TypedDict, total=False):
    """Type definition for model configuration."""

    # Architecture
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int

    # Activation and normalization
    hidden_act: str
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    layer_norm_eps: float

    # Position and embeddings
    max_position_embeddings: int
    type_vocab_size: int

    # Advanced configuration
    initializer_range: float
    use_cache: bool
    rope_theta: float
    rope_scaling: Optional[Dict[str, Any]]
    use_bias: bool
    tie_word_embeddings: bool


class DataConfig(TypedDict, total=False):
    """Type definition for data configuration."""

    # Data paths
    data_path: str
    train_file: Optional[str]
    validation_file: Optional[str]

    # Data processing
    max_seq_length: int
    chunk_size: int
    packed_sequences: bool

    # Tokenization (FinPileTokenizers integration)
    tokenizer_name: Optional[str]  # Deprecated but kept for compatibility
    eod_token_id: Optional[int]

    # Data curation (DataDecide)
    proxy_model_size: Optional[str]
    num_proxy_steps: Optional[int]
    eval_metrics: Optional[List[str]]

    # Preprocessing
    preprocessing_num_workers: Optional[int]
    streaming: Optional[bool]


class ExperimentConfig(TypedDict):
    """Type definition for complete experiment configuration."""

    # Core configuration sections
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    # Experiment metadata
    experiment_name: str
    output_dir: str
    run_name: Optional[str]

    # Hardware configuration
    device: Optional[str]
    num_gpus: Optional[int]
    distributed: Optional[bool]

    # Checkpointing and resuming
    resume_from_checkpoint: Optional[str]
    save_total_limit: Optional[int]

    # Wandb and logging
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    log_level: Optional[str]


# =============================================================================
# Data Structure TypedDict Classes
# =============================================================================


class BatchData(TypedDict):
    """Type definition for training/evaluation batch data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    # Optional fields for packed sequences
    position_ids: Optional[torch.Tensor]
    packed_mask: Optional[torch.Tensor]


class EvaluationMetrics(TypedDict):
    """Type definition for evaluation metrics."""

    loss: float
    perplexity: float
    tokens: int

    # Optional additional metrics
    accuracy: Optional[float]
    bleu_score: Optional[float]
    rouge_scores: Optional[Dict[str, float]]


class CheckpointState(TypedDict):
    """Type definition for checkpoint state."""

    # Training state
    epoch: int
    global_step: int

    # Model and optimizer states
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    lr_scheduler_state_dict: Dict[str, Any]

    # Configuration
    config: ExperimentConfig

    # Metrics tracking
    best_eval_loss: float
    train_loss_history: List[float]
    eval_loss_history: List[float]


class DatasetStatistics(TypedDict):
    """Type definition for dataset statistics from DataDecide."""

    total_documents: float
    avg_length: float
    std_length: float
    vocabulary_size: float
    deduplication_rate: float
    quality_score: float


# =============================================================================
# Protocol Interfaces
# =============================================================================


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer interface compatibility."""

    def encode(self, text: str, **kwargs: Any) -> List[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, token_ids: List[int], **kwargs: Any) -> str:
        """Decode token IDs to text."""
        ...

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...


class DatasetProtocol(Protocol):
    """Protocol for dataset interface compatibility."""

    def __len__(self) -> int:
        """Get dataset length."""
        ...

    def __getitem__(self, idx: int) -> BatchData:
        """Get item by index."""
        ...


class ModelProtocol(Protocol):
    """Protocol for model interface compatibility."""

    def forward(self, **kwargs: Any) -> Any:
        """Forward pass through the model."""
        ...

    def train(self, mode: bool = True) -> Any:
        """Set model to training mode."""
        ...

    def eval(self) -> Any:
        """Set model to evaluation mode."""
        ...

    def parameters(self) -> Any:
        """Get model parameters."""
        ...

    def named_parameters(self) -> Any:
        """Get named model parameters."""
        ...


class DataLoaderProtocol(Protocol):
    """Protocol for data loader interface compatibility."""

    def __iter__(self) -> Any:
        """Iterate over batches."""
        ...

    def __len__(self) -> int:
        """Get number of batches."""
        ...

    @property
    def dataset(self) -> DatasetProtocol:
        """Get underlying dataset."""
        ...


class OptimizerProtocol(Protocol):
    """Protocol for optimizer interface compatibility."""

    def step(self) -> None:
        """Perform optimization step."""
        ...

    def zero_grad(self) -> None:
        """Zero gradients."""
        ...

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups."""
        ...


class SchedulerProtocol(Protocol):
    """Protocol for learning rate scheduler interface compatibility."""

    def step(self) -> None:
        """Update learning rate."""
        ...

    def get_last_lr(self) -> List[float]:
        """Get last learning rate."""
        ...


# =============================================================================
# Validation Functions
# =============================================================================


def validate_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """Validate and convert training configuration with proper defaults."""

    required_fields = ["model_size", "batch_size", "learning_rate", "max_steps"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required training config field: {field}")

    # Set sensible defaults for optional fields
    defaults = {
        "per_device_eval_batch_size": config.get("batch_size", 8),
        "num_train_epochs": 3,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "linear",
        "warmup_steps": 500,
        "fp16": False,
        "num_workers": 4,
        "logging_steps": 100,
        "eval_steps": 500,
        "save_steps": 1000,
        "report_to": ["wandb"],
    }

    # Merge defaults with provided config
    validated_config = {**defaults, **config}

    return validated_config  # type: ignore


def validate_model_config(config: Dict[str, Any]) -> ModelConfig:
    """Validate and convert model configuration."""

    required_fields = ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required model config field: {field}")

    # Validate head dimensions
    if config["hidden_size"] % config["num_attention_heads"] != 0:
        raise ValueError(
            f"hidden_size ({config['hidden_size']}) must be divisible by "
            f"num_attention_heads ({config['num_attention_heads']})"
        )

    return config  # type: ignore


def validate_experiment_config(config: Dict[str, Any]) -> ExperimentConfig:
    """Validate complete experiment configuration."""

    if "training" not in config:
        raise ValueError("Missing 'training' section in experiment config")

    if "model" not in config:
        raise ValueError("Missing 'model' section in experiment config")

    if "data" not in config:
        raise ValueError("Missing 'data' section in experiment config")

    # Validate subsections
    config["training"] = validate_training_config(config["training"])
    config["model"] = validate_model_config(config["model"])

    # Set defaults for experiment-level settings
    if "experiment_name" not in config:
        config["experiment_name"] = f"olmo-{config['training']['model_size']}"

    if "output_dir" not in config:
        config["output_dir"] = "./outputs"

    return config  # type: ignore


# =============================================================================
# Helper Type Guards
# =============================================================================


def is_batch_data(data: Any) -> bool:
    """Check if data conforms to BatchData structure."""
    return (
        isinstance(data, dict)
        and "input_ids" in data
        and "attention_mask" in data
        and "labels" in data
        and isinstance(data["input_ids"], torch.Tensor)
        and isinstance(data["attention_mask"], torch.Tensor)
        and isinstance(data["labels"], torch.Tensor)
    )


def is_metrics_dict(metrics: Any) -> bool:
    """Check if metrics conforms to MetricsDict structure."""
    return isinstance(metrics, dict) and all(
        isinstance(k, str) and isinstance(v, (int, float)) for k, v in metrics.items()
    )
