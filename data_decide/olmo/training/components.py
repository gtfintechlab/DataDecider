"""
Training component managers following Single Responsibility Principle.

This module provides specialized managers for different aspects of model training,
breaking down the monolithic trainer into focused, testable components.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import wandb
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from ...utils.logging_utils import get_logger
from ...utils.type_definitions import (
    DatasetProtocol,
    ExperimentConfig,
    MetricsDict,
    TrainingConfig,
)

logger = get_logger(__name__)


# =============================================================================
# Abstract Base Classes for Component Interfaces
# =============================================================================


class TrainingComponent(ABC):
    """Abstract base class for training components."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.training_config: TrainingConfig = config["training"]


class ModelManager(TrainingComponent):
    """Manages model creation, configuration, and lifecycle."""

    def __init__(self, config: ExperimentConfig, accelerator: Accelerator) -> None:
        super().__init__(config)
        self.accelerator = accelerator
        self._model: Optional[torch.nn.Module] = None

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        """Create and return a model instance."""
        pass

    @abstractmethod
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepare model for training with accelerator."""
        pass

    @property
    def model(self) -> torch.nn.Module:
        """Get the current model instance."""
        if self._model is None:
            self._model = self.create_model()
            self._model = self.prepare_model(self._model)
        return self._model


class DataManager(TrainingComponent):
    """Manages data loading, sampling, and data loader configuration."""

    def __init__(
        self,
        config: ExperimentConfig,
        accelerator: Accelerator,
        train_dataset: Optional[DatasetProtocol] = None,
        eval_dataset: Optional[DatasetProtocol] = None,
    ) -> None:
        super().__init__(config)
        self.accelerator = accelerator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._train_dataloader: Optional[DataLoader] = None
        self._eval_dataloader: Optional[DataLoader] = None

    def create_train_dataloader(self) -> DataLoader:
        """Create training data loader with proper sampling strategy."""
        if self.train_dataset is None:
            raise ValueError("Training dataset is required")

        # Setup distributed sampling if needed
        train_sampler: Optional[DistributedSampler] = None
        if self.accelerator.num_processes > 1:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
            )

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config["batch_size"],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.training_config["num_workers"],
            pin_memory=True,
        )

        return self.accelerator.prepare(dataloader)

    def create_eval_dataloader(self) -> Optional[DataLoader]:
        """Create evaluation data loader."""
        if self.eval_dataset is None:
            return None

        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.training_config["per_device_eval_batch_size"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
            pin_memory=True,
        )

        return self.accelerator.prepare(dataloader)

    @property
    def train_dataloader(self) -> DataLoader:
        """Get the training data loader, creating it if necessary."""
        if self._train_dataloader is None:
            self._train_dataloader = self.create_train_dataloader()
        return self._train_dataloader

    @property
    def eval_dataloader(self) -> Optional[DataLoader]:
        """Get the evaluation data loader, creating it if necessary."""
        if self._eval_dataloader is None:
            self._eval_dataloader = self.create_eval_dataloader()
        return self._eval_dataloader


class OptimizationManager(TrainingComponent):
    """Manages optimizer and learning rate scheduler creation and configuration."""

    def __init__(self, config: ExperimentConfig, model: torch.nn.Module, accelerator: Accelerator) -> None:
        super().__init__(config)
        self.model = model
        self.accelerator = accelerator
        self._optimizer: Optional[Optimizer] = None
        self._scheduler: Optional[LRScheduler] = None

    def create_optimizer(self) -> Optimizer:
        """Create optimizer with weight decay parameter grouping."""
        # Parameters that should not have weight decay
        no_decay_params = [
            "bias",
            "layer_norm.weight",
            "ln_1.weight",
            "ln_2.weight",
            "norm.weight",
        ]

        # Group parameters based on weight decay policy
        optimizer_grouped_parameters: List[Dict[str, Any]] = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_params)],
                "weight_decay": self.training_config["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_params)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.training_config["learning_rate"],
            betas=(
                self.training_config["adam_beta1"],
                self.training_config["adam_beta2"],
            ),
            eps=self.training_config["adam_epsilon"],
        )

        return self.accelerator.prepare(optimizer)

    def create_scheduler(self, num_training_steps: int) -> LRScheduler:
        """Create learning rate scheduler based on configuration."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be created before scheduler")

        warmup_steps: int = self.training_config["warmup_steps"]
        scheduler_type = self.training_config["lr_scheduler_type"]

        if scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return self.accelerator.prepare(scheduler)

    @property
    def optimizer(self) -> Optimizer:
        """Get the optimizer, creating it if necessary."""
        if self._optimizer is None:
            self._optimizer = self.create_optimizer()
        return self._optimizer

    def get_scheduler(self, num_training_steps: int) -> LRScheduler:
        """Get the learning rate scheduler, creating it if necessary."""
        if self._scheduler is None:
            self._scheduler = self.create_scheduler(num_training_steps)
        return self._scheduler


class LoggingManager(TrainingComponent):
    """Manages experiment logging, tracking, and metric reporting."""

    def __init__(self, config: ExperimentConfig, accelerator: Accelerator) -> None:
        super().__init__(config)
        self.accelerator = accelerator
        self.output_dir: str = config.get("output_dir", "./outputs")
        self._initialized = False

    def setup_logging(self) -> None:
        """Setup logging infrastructure for training."""
        if not self.accelerator.is_main_process:
            return

        # Initialize wandb if configured
        if "wandb" in self.training_config["report_to"]:
            wandb.init(
                project="olmo-training",
                name=f"olmo-{self.training_config['model_size']}",
                config=self.config,
            )

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Save configuration
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        self._initialized = True
        logger.info(f"Logging initialized. Output directory: {self.output_dir}")

    def log_metrics(self, metrics: MetricsDict, step: int) -> None:
        """Log metrics to configured tracking services."""
        if not self.accelerator.is_main_process:
            return

        if not self._initialized:
            self.setup_logging()

        # Log to wandb
        if "wandb" in self.training_config["report_to"]:
            wandb.log(metrics, step=step)

        # Log to console
        logger.info(f"Step {step}: {metrics}")

    def finish_logging(self) -> None:
        """Clean up logging resources."""
        if self.accelerator.is_main_process and "wandb" in self.training_config["report_to"]:
            wandb.finish()


class CheckpointManager(TrainingComponent):
    """Manages model checkpointing, saving, and loading."""

    def __init__(self, config: ExperimentConfig, accelerator: Accelerator, output_dir: str) -> None:
        super().__init__(config)
        self.accelerator = accelerator
        self.output_dir = output_dir

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        global_step: int,
        epoch: int = 0,
        additional_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a training checkpoint."""
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Prepare checkpoint state
        checkpoint_state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.accelerator.unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": self.config,
        }

        # Add any additional state
        if additional_state:
            checkpoint_state.update(additional_state)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        torch.save(checkpoint_state, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def save_model(self, model: torch.nn.Module, output_dir: str) -> None:
        """Save the model for inference."""
        if not self.accelerator.is_main_process:
            return

        os.makedirs(output_dir, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)

        logger.info(f"Saved model to {output_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint from disk."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return torch.load(checkpoint_path, map_location="cpu")


class EvaluationManager(TrainingComponent):
    """Manages model evaluation and metric calculation."""

    def __init__(self, config: ExperimentConfig, accelerator: Accelerator) -> None:
        super().__init__(config)
        self.accelerator = accelerator

    def evaluate(self, model: torch.nn.Module, eval_dataloader: DataLoader) -> MetricsDict:
        """Run evaluation and return metrics."""
        logger.info("Running evaluation...")
        model.eval()

        total_loss: float = 0.0
        total_tokens: int = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                # Validate batch structure
                if not self._is_valid_batch(batch):
                    continue

                outputs = model(**batch)
                loss = outputs.loss

                # Accumulate metrics
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_tokens += batch["attention_mask"].sum().item()

        # Calculate final metrics
        avg_loss: float = total_loss / len(eval_dataloader.dataset)
        perplexity: float = torch.exp(torch.tensor(avg_loss)).item()

        model.train()

        return {"loss": avg_loss, "perplexity": perplexity, "tokens": total_tokens}

    def _is_valid_batch(self, batch: Any) -> bool:
        """Validate batch structure for evaluation."""
        required_keys = ["input_ids", "attention_mask", "labels"]
        return (
            isinstance(batch, dict)
            and all(key in batch for key in required_keys)
            and all(isinstance(batch[key], torch.Tensor) for key in required_keys)
        )


# =============================================================================
# Concrete Implementation for OLMo Models
# =============================================================================


class OLMoModelManager(ModelManager):
    """Model manager specifically for OLMo models."""

    def create_model(self) -> torch.nn.Module:
        """Create an OLMo model based on configuration."""
        from ..models.configuration_olmo import OLMO_CONFIGS
        from ..models.olmo_model import OLMoForCausalLM

        model_config = OLMO_CONFIGS[self.training_config["model_size"]]
        return OLMoForCausalLM(model_config)

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepare OLMo model for distributed training."""
        return self.accelerator.prepare(model)
