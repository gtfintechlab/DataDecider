"""
Refactored OLMo trainer following Single Responsibility Principle.

This module provides a clean, modular trainer that orchestrates specialized
components rather than handling all concerns directly.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from accelerate import Accelerator
from tqdm import tqdm

from ...utils.logging_utils import get_logger
from ...utils.type_definitions import (
    BatchData,
    ConfigDict,
    DatasetProtocol,
    ExperimentConfig,
    MetricsDict,
    TrainingConfig,
    is_batch_data,
    validate_experiment_config,
)
from .components import (
    CheckpointManager,
    DataManager,
    EvaluationManager,
    LoggingManager,
    OLMoModelManager,
    OptimizationManager,
)

logger = get_logger(__name__)


class OLMoTrainer:
    """
    Refactored OLMo trainer following Single Responsibility Principle.

    This trainer orchestrates specialized component managers:
    - ModelManager: Handles model creation and preparation
    - DataManager: Manages data loading and sampling
    - OptimizationManager: Creates optimizers and schedulers
    - LoggingManager: Handles experiment tracking
    - CheckpointManager: Manages model saving/loading
    - EvaluationManager: Runs evaluation and metrics

    The trainer's sole responsibility is orchestrating the training loop
    and coordinating between components.
    """

    def __init__(
        self,
        config: Union[ConfigDict, ExperimentConfig],
        model: Optional[torch.nn.Module] = None,
        train_dataset: Optional[DatasetProtocol] = None,
        eval_dataset: Optional[DatasetProtocol] = None,
        tokenizer: Optional[Any] = None,  # Kept for interface compatibility
    ) -> None:
        """
        Initialize the refactored trainer with component managers.

        Args:
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            model: Pre-initialized model (optional)
        """
        # Validate and store configuration
        self.config: ExperimentConfig = validate_experiment_config(config)
        self.training_config: TrainingConfig = self.config["training"]

        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision="fp16" if self.training_config["fp16"] else "no",
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            log_with=self.training_config["report_to"],
        )

        # Store tokenizer for interface compatibility (not used in practice)
        self.tokenizer = tokenizer

        # Initialize component managers
        self._setup_component_managers(train_dataset, eval_dataset, model)

        # Training state
        self.global_step: int = 0
        self.best_eval_loss: float = float("inf")

    def _setup_component_managers(
        self,
        train_dataset: Optional[DatasetProtocol],
        eval_dataset: Optional[DatasetProtocol],
        model: Optional[torch.nn.Module],
    ) -> None:
        """Initialize all component managers."""
        # Model management
        self.model_manager = OLMoModelManager(self.config, self.accelerator)
        if model is not None:
            self.model_manager._model = self.accelerator.prepare(model)

        # Data management
        self.data_manager = DataManager(self.config, self.accelerator, train_dataset, eval_dataset)

        # Get model for optimization setup
        model_for_optim = self.model_manager.model

        # Optimization management
        self.optimization_manager = OptimizationManager(self.config, model_for_optim, self.accelerator)

        # Logging management
        self.logging_manager = LoggingManager(self.config, self.accelerator)

        # Checkpoint management
        output_dir = self.config.get("output_dir", "./outputs")
        self.checkpoint_manager = CheckpointManager(self.config, self.accelerator, output_dir)

        # Evaluation management
        self.evaluation_manager = EvaluationManager(self.config, self.accelerator)

    def train(self) -> None:
        """
        Main training loop - orchestrates all components.

        The trainer's primary responsibility: coordinate the training process
        by delegating specific tasks to appropriate component managers.
        """
        logger.info(f"Starting training for {self.training_config['model_size']} model")
        logger.info(f"Total training steps: {self.training_config['max_steps']}")

        # Setup components
        self._prepare_training_components()

        # Initialize logging
        self.logging_manager.setup_logging()

        # Training metrics
        total_loss: float = 0.0

        # Training loop with progress tracking
        progress_bar = tqdm(
            total=self.training_config["max_steps"],
            disable=not self.accelerator.is_local_main_process,
        )

        try:
            for epoch in range(self.training_config["num_train_epochs"]):
                self.model_manager.model.train()

                for step, batch in enumerate(self.data_manager.train_dataloader):
                    # Validate batch structure
                    if not is_batch_data(batch):
                        logger.warning(f"Invalid batch structure at step {step}")
                        continue

                    # Forward pass
                    loss = self._training_step(batch)
                    total_loss += loss

                    # Gradient accumulation and optimization
                    if self._should_update_weights(step):
                        self._optimization_step()
                        self.global_step += 1
                        progress_bar.update(1)

                        # Logging
                        if self._should_log():
                            self._log_training_metrics(total_loss)
                            total_loss = 0.0

                        # Evaluation
                        if self._should_evaluate():
                            self._run_evaluation()

                        # Checkpointing
                        if self._should_save_checkpoint():
                            self._save_checkpoint()

                        # Check completion
                        if self.global_step >= self.training_config["max_steps"]:
                            logger.info("Reached max steps. Stopping training.")
                            return

            logger.info("Training completed!")

        finally:
            # Cleanup
            progress_bar.close()
            self.logging_manager.finish_logging()

    def _prepare_training_components(self) -> None:
        """Prepare all training components."""
        # Calculate training steps for scheduler
        train_dataloader = self.data_manager.train_dataloader
        num_training_steps = len(train_dataloader) * self.training_config["num_train_epochs"]
        if self.training_config["max_steps"] > 0:
            num_training_steps = min(num_training_steps, self.training_config["max_steps"])

        # Initialize optimizer and scheduler
        _ = self.optimization_manager.optimizer  # Trigger creation
        self.lr_scheduler = self.optimization_manager.get_scheduler(num_training_steps)

    def _training_step(self, batch: BatchData) -> float:
        """Execute a single training step."""
        outputs = self.model_manager.model(**batch)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        loss = loss / self.training_config["gradient_accumulation_steps"]

        # Backward pass
        self.accelerator.backward(loss)

        return loss.item()

    def _optimization_step(self) -> None:
        """Execute optimization step with gradient clipping."""
        # Gradient clipping
        if self.training_config["max_grad_norm"] > 0:
            self.accelerator.clip_grad_norm_(
                self.model_manager.model.parameters(),
                self.training_config["max_grad_norm"],
            )

        # Optimizer step
        self.optimization_manager.optimizer.step()
        self.lr_scheduler.step()
        self.optimization_manager.optimizer.zero_grad()

    def _should_update_weights(self, step: int) -> bool:
        """Check if weights should be updated (gradient accumulation)."""
        return (step + 1) % self.training_config["gradient_accumulation_steps"] == 0

    def _should_log(self) -> bool:
        """Check if metrics should be logged."""
        return self.global_step % self.training_config["logging_steps"] == 0

    def _should_evaluate(self) -> bool:
        """Check if evaluation should be run."""
        return (
            self.global_step % self.training_config["eval_steps"] == 0 and self.data_manager.eval_dataloader is not None
        )

    def _should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved."""
        return self.global_step % self.training_config["save_steps"] == 0

    def _log_training_metrics(self, total_loss: float) -> None:
        """Log training metrics."""
        avg_loss = total_loss / self.training_config["logging_steps"]
        metrics = {
            "train/loss": avg_loss,
            "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
            "train/global_step": self.global_step,
        }
        self.logging_manager.log_metrics(metrics, self.global_step)

    def _run_evaluation(self) -> None:
        """Run evaluation and log results."""
        eval_dataloader = self.data_manager.eval_dataloader
        if eval_dataloader is None:
            return

        eval_metrics = self.evaluation_manager.evaluate(self.model_manager.model, eval_dataloader)

        # Log evaluation metrics
        eval_metrics_with_prefix = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logging_manager.log_metrics(eval_metrics_with_prefix, self.global_step)

        # Save best model
        if eval_metrics["loss"] < self.best_eval_loss:
            self.best_eval_loss = eval_metrics["loss"]
            best_model_dir = f"{self.checkpoint_manager.output_dir}/best_model"
            self.checkpoint_manager.save_model(self.model_manager.model, best_model_dir)

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        additional_state = {
            "best_eval_loss": self.best_eval_loss,
        }

        self.checkpoint_manager.save_checkpoint(
            model=self.model_manager.model,
            optimizer=self.optimization_manager.optimizer,
            scheduler=self.lr_scheduler,
            global_step=self.global_step,
            additional_state=additional_state,
        )

    # =============================================================================
    # Public API Methods
    # =============================================================================

    def evaluate(self) -> MetricsDict:
        """Run evaluation and return metrics."""
        eval_dataloader = self.data_manager.eval_dataloader
        if eval_dataloader is None:
            raise ValueError("No evaluation dataset provided")

        return self.evaluation_manager.evaluate(self.model_manager.model, eval_dataloader)

    def save_model(self, output_dir: str) -> None:
        """Save the trained model."""
        self.checkpoint_manager.save_model(self.model_manager.model, output_dir)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a training checkpoint."""
        checkpoint_state = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        # Restore model state
        self.model_manager.model.load_state_dict(checkpoint_state["model_state_dict"])

        # Restore optimizer state
        self.optimization_manager.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

        # Restore scheduler state
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])

        # Restore training state
        self.global_step = checkpoint_state["global_step"]
        self.best_eval_loss = checkpoint_state.get("best_eval_loss", float("inf"))

        logger.info(f"Loaded checkpoint from step {self.global_step}")
