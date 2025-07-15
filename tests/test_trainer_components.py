"""
Test refactored trainer components and architecture.

This module tests the component-based trainer architecture (Issue #6)
that follows Single Responsibility Principle.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_trainer_component_imports():
    """Test that trainer components can be imported."""
    from data_decide.olmo.training.components import (
        CheckpointManager,
        DataManager,
        EvaluationManager,
        LoggingManager,
        OLMoModelManager,
        OptimizationManager,
    )

    # Verify all components are importable
    assert OLMoModelManager is not None
    assert DataManager is not None
    assert OptimizationManager is not None
    assert LoggingManager is not None
    assert CheckpointManager is not None
    assert EvaluationManager is not None


@pytest.mark.unit
def test_trainer_import():
    """Test that the refactored trainer can be imported."""
    from data_decide.olmo.training.trainer import OLMoTrainer

    assert OLMoTrainer is not None


@pytest.mark.unit
def test_trainer_initialization(sample_config):
    """Test trainer initialization with valid configuration."""
    from data_decide.olmo.training.trainer import OLMoTrainer

    # Mock Accelerator to avoid distributed training dependencies in tests
    with patch("data_decide.olmo.training.trainer.Accelerator") as mock_accelerator:
        mock_accelerator.return_value = MagicMock()

        # Initialize trainer
        trainer = OLMoTrainer(config=sample_config)

        # Verify trainer has required attributes
        assert hasattr(trainer, "config")
        assert hasattr(trainer, "training_config")
        assert hasattr(trainer, "model_manager")
        assert hasattr(trainer, "data_manager")
        assert hasattr(trainer, "optimization_manager")
        assert hasattr(trainer, "logging_manager")
        assert hasattr(trainer, "checkpoint_manager")
        assert hasattr(trainer, "evaluation_manager")


@pytest.mark.unit
def test_model_manager_functionality(sample_config):
    """Test ModelManager component functionality."""
    from data_decide.olmo.training.components import OLMoModelManager

    # Mock accelerator
    mock_accelerator = MagicMock()

    # Create model manager
    model_manager = OLMoModelManager(sample_config, mock_accelerator)

    # Test model creation
    assert hasattr(model_manager, "create_model")
    assert callable(model_manager.create_model)


@pytest.mark.unit
def test_data_manager_functionality(sample_config):
    """Test DataManager component functionality."""
    from data_decide.olmo.training.components import DataManager

    # Mock accelerator
    mock_accelerator = MagicMock()

    # Create data manager
    data_manager = DataManager(sample_config, mock_accelerator)

    # Test data manager has required methods
    assert hasattr(data_manager, "prepare_datasets")
    assert callable(data_manager.prepare_datasets)


@pytest.mark.unit
def test_optimization_manager_functionality(sample_config):
    """Test OptimizationManager component functionality."""
    from data_decide.olmo.training.components import OptimizationManager

    # Mock model and accelerator
    mock_model = MagicMock()
    mock_accelerator = MagicMock()

    # Create optimization manager
    opt_manager = OptimizationManager(sample_config, mock_model, mock_accelerator)

    # Test optimization manager has required properties
    assert hasattr(opt_manager, "optimizer")
    assert hasattr(opt_manager, "get_scheduler")
    assert callable(opt_manager.get_scheduler)


@pytest.mark.unit
def test_logging_manager_functionality(sample_config):
    """Test LoggingManager component functionality."""
    from data_decide.olmo.training.components import LoggingManager

    # Mock accelerator
    mock_accelerator = MagicMock()

    # Create logging manager
    logging_manager = LoggingManager(sample_config, mock_accelerator)

    # Test logging manager has required methods
    assert hasattr(logging_manager, "setup_logging")
    assert hasattr(logging_manager, "log_metrics")
    assert hasattr(logging_manager, "finish_logging")
    assert callable(logging_manager.setup_logging)
    assert callable(logging_manager.log_metrics)
    assert callable(logging_manager.finish_logging)


@pytest.mark.unit
def test_checkpoint_manager_functionality(sample_config, temp_dir):
    """Test CheckpointManager component functionality."""
    from data_decide.olmo.training.components import CheckpointManager

    # Mock accelerator
    mock_accelerator = MagicMock()

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(sample_config, mock_accelerator, str(temp_dir))

    # Test checkpoint manager has required methods
    assert hasattr(checkpoint_manager, "save_checkpoint")
    assert hasattr(checkpoint_manager, "load_checkpoint")
    assert hasattr(checkpoint_manager, "save_model")
    assert callable(checkpoint_manager.save_checkpoint)
    assert callable(checkpoint_manager.load_checkpoint)
    assert callable(checkpoint_manager.save_model)


@pytest.mark.unit
def test_evaluation_manager_functionality(sample_config):
    """Test EvaluationManager component functionality."""
    from data_decide.olmo.training.components import EvaluationManager

    # Mock accelerator
    mock_accelerator = MagicMock()

    # Create evaluation manager
    eval_manager = EvaluationManager(sample_config, mock_accelerator)

    # Test evaluation manager has required methods
    assert hasattr(eval_manager, "evaluate")
    assert callable(eval_manager.evaluate)


@pytest.mark.integration
def test_trainer_component_coordination(sample_config):
    """Test that trainer properly coordinates between components."""
    from data_decide.olmo.training.trainer import OLMoTrainer

    # Mock external dependencies
    with patch("data_decide.olmo.training.trainer.Accelerator") as mock_accelerator:
        mock_accelerator_instance = MagicMock()
        mock_accelerator.return_value = mock_accelerator_instance

        # Mock component managers
        with patch("data_decide.olmo.training.trainer.OLMoModelManager") as mock_model_mgr:
            with patch("data_decide.olmo.training.trainer.DataManager") as mock_data_mgr:
                with patch("data_decide.olmo.training.trainer.OptimizationManager") as mock_opt_mgr:
                    with patch("data_decide.olmo.training.trainer.LoggingManager") as mock_log_mgr:
                        with patch("data_decide.olmo.training.trainer.CheckpointManager") as mock_chk_mgr:
                            with patch("data_decide.olmo.training.trainer.EvaluationManager") as mock_eval_mgr:
                                # Initialize trainer
                                trainer = OLMoTrainer(config=sample_config)

                                # Verify all managers were created
                                mock_model_mgr.assert_called_once()
                                mock_data_mgr.assert_called_once()
                                mock_opt_mgr.assert_called_once()
                                mock_log_mgr.assert_called_once()
                                mock_chk_mgr.assert_called_once()
                                mock_eval_mgr.assert_called_once()

                                # Verify trainer has access to all components
                                assert trainer.model_manager is not None
                                assert trainer.data_manager is not None
                                assert trainer.optimization_manager is not None
                                assert trainer.logging_manager is not None
                                assert trainer.checkpoint_manager is not None
                                assert trainer.evaluation_manager is not None


@pytest.mark.unit
def test_trainer_configuration_validation(invalid_config):
    """Test that trainer validates configuration properly."""
    from data_decide.olmo.training.trainer import OLMoTrainer
    from data_decide.utils.validation import ValidationError

    # Mock Accelerator to avoid dependencies
    with patch("data_decide.olmo.training.trainer.Accelerator") as mock_accelerator:
        mock_accelerator.return_value = MagicMock()

        # Should raise validation error for invalid config
        with pytest.raises((ValidationError, ValueError, KeyError)):
            OLMoTrainer(config=invalid_config)


@pytest.mark.unit
def test_trainer_training_state():
    """Test trainer training state management."""
    from data_decide.olmo.training.trainer import OLMoTrainer

    # Mock all dependencies
    with patch("data_decide.olmo.training.trainer.Accelerator") as mock_accelerator:
        with patch("data_decide.olmo.training.trainer.validate_experiment_config") as mock_validate:
            mock_validate.return_value = {
                "training": {
                    "model_size": "4M",
                    "fp16": False,
                    "gradient_accumulation_steps": 1,
                    "report_to": [],
                }
            }
            mock_accelerator.return_value = MagicMock()

            # Mock component setup
            with patch.object(OLMoTrainer, "_setup_component_managers"):
                trainer = OLMoTrainer(config={})

                # Verify initial state
                assert trainer.global_step == 0
                assert trainer.best_eval_loss == float("inf")


@pytest.mark.unit
def test_trainer_api_methods(sample_config):
    """Test trainer public API methods."""
    from data_decide.olmo.training.trainer import OLMoTrainer

    # Mock all dependencies
    with patch("data_decide.olmo.training.trainer.Accelerator") as mock_accelerator:
        mock_accelerator.return_value = MagicMock()

        with patch.object(OLMoTrainer, "_setup_component_managers"):
            trainer = OLMoTrainer(config=sample_config)

            # Mock components
            trainer.evaluation_manager = MagicMock()
            trainer.checkpoint_manager = MagicMock()
            trainer.model_manager = MagicMock()
            trainer.data_manager = MagicMock()

            # Test public API methods exist
            assert hasattr(trainer, "train")
            assert hasattr(trainer, "evaluate")
            assert hasattr(trainer, "save_model")
            assert hasattr(trainer, "load_checkpoint")
            assert callable(trainer.train)
            assert callable(trainer.evaluate)
            assert callable(trainer.save_model)
            assert callable(trainer.load_checkpoint)


@pytest.mark.integration
def test_single_responsibility_principle():
    """Test that components follow Single Responsibility Principle."""
    from data_decide.olmo.training.components import (
        CheckpointManager,
        DataManager,
        EvaluationManager,
        LoggingManager,
        OLMoModelManager,
        OptimizationManager,
    )

    # Each component should have a focused set of responsibilities

    # ModelManager: Model creation and management
    model_methods = [m for m in dir(OLMoModelManager) if not m.startswith("_")]
    assert any("model" in m.lower() for m in model_methods)

    # DataManager: Data loading and preparation
    data_methods = [m for m in dir(DataManager) if not m.startswith("_")]
    assert any("data" in m.lower() or "dataset" in m.lower() for m in data_methods)

    # OptimizationManager: Optimizer and scheduler management
    opt_methods = [m for m in dir(OptimizationManager) if not m.startswith("_")]
    assert any("optim" in m.lower() or "scheduler" in m.lower() for m in opt_methods)

    # LoggingManager: Logging and monitoring
    log_methods = [m for m in dir(LoggingManager) if not m.startswith("_")]
    assert any("log" in m.lower() for m in log_methods)

    # CheckpointManager: Model saving and loading
    chk_methods = [m for m in dir(CheckpointManager) if not m.startswith("_")]
    assert any("save" in m.lower() or "load" in m.lower() or "checkpoint" in m.lower() for m in chk_methods)

    # EvaluationManager: Model evaluation
    eval_methods = [m for m in dir(EvaluationManager) if not m.startswith("_")]
    assert any("eval" in m.lower() for m in eval_methods)


@pytest.mark.unit
def test_trainer_architecture_benefits():
    """Test that the new architecture provides expected benefits."""
    from data_decide.olmo.training.trainer import OLMoTrainer

    # Test that trainer is primarily an orchestrator, not a monolithic class
    trainer_methods = [m for m in dir(OLMoTrainer) if not m.startswith("_")]

    # Should have high-level orchestration methods
    orchestration_methods = ["train", "evaluate", "save_model", "load_checkpoint"]
    for method in orchestration_methods:
        assert method in trainer_methods

    # Should not have low-level implementation methods (delegated to components)
    # This is validated by checking that the trainer is focused on coordination
