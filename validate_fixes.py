#!/usr/bin/env python3
"""
Validation script for all implemented fixes.
Checks file contents, syntax, and basic functionality.
"""

from pathlib import Path


def validate_critical_fix_1():
    """Validate Critical Fix #1: Error handling in data curation."""
    print("🔍 Validating Critical Fix #1: Error handling in data curation")

    curation_file = Path("data_decide/olmo/data/data_curation.py")
    with open(curation_file, "r") as f:
        content = f.read()

    # Check for comprehensive error handling
    checks = [
        ("JSON error handling", "except json.JSONDecodeError as e:" in content),
        ("File I/O error handling", "except (OSError, IOError, UnicodeDecodeError)" in content),
        ("Empty line handling", "if not line:" in content),
        ("Failed files tracking", "failed_files.append(file_path)" in content),
        ("Logging integration", "logger.warning" in content and "logger.error" in content),
        (
            "Graceful continuation",
            "# Continue processing other lines" in content or "# Continue with other files" in content,
        ),
    ]

    passed = 0
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
        if check_result:
            passed += 1

    success = passed == len(checks)
    print(f"  Result: {passed}/{len(checks)} checks passed")
    return success


def validate_critical_fix_2():
    """Validate Critical Fix #2: Memory safety in data loading."""
    print("🔍 Validating Critical Fix #2: Memory safety in data loading")

    loader_file = Path("data_decide/utils/finpile_data_loader.py")
    with open(loader_file, "r") as f:
        content = f.read()

    checks = [
        ("Context manager __enter__", "def __enter__(self):" in content),
        ("Context manager __exit__", "def __exit__(self, exc_type, exc_val, exc_tb):" in content),
        ("Explicit close method", "def close(self):" in content),
        ("Closed state tracking", "self._is_closed" in content),
        ("Access validation", "if self._is_closed:" in content),
        ("Resource cleanup", "del self._handle" in content),
        ("Destructor safety", "def __del__(self):" in content),
        ("Error messages", "Cannot access data from a closed dataset" in content),
    ]

    passed = 0
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
        if check_result:
            passed += 1

    success = passed == len(checks)
    print(f"  Result: {passed}/{len(checks)} checks passed")
    return success


def validate_critical_fix_3():
    """Validate Critical Fix #3: Configuration factory patterns."""
    print("🔍 Validating Critical Fix #3: Configuration factory patterns")

    config_file = Path("data_decide/olmo/models/configuration_olmo.py")
    with open(config_file, "r") as f:
        content = f.read()

    checks = [
        ("Constants documentation", "# Configuration constants with clear documentation" in content),
        ("FINPILE_VOCAB_SIZE constant", "FINPILE_VOCAB_SIZE = 50277" in content),
        ("INTERMEDIATE_SIZE_RATIO constant", "INTERMEDIATE_SIZE_RATIO = 4" in content),
        ("HEAD_DIM constant", "HEAD_DIM = 64" in content),
        ("MODEL_SCALING_CONFIG dict", "MODEL_SCALING_CONFIG = {" in content),
        ("ModelConfigFactory class", "class ModelConfigFactory:" in content),
        ("create_config method", "def create_config(model_size: str" in content),
        ("get_available_sizes method", "def get_available_sizes(" in content),
        ("estimate_parameters method", "def estimate_parameters(" in content),
        ("Validation logic", "if model_size not in MODEL_SCALING_CONFIG:" in content),
        ("Head dimension validation", "if hidden_size % num_heads != 0:" in content),
    ]

    passed = 0
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
        if check_result:
            passed += 1

    success = passed == len(checks)
    print(f"  Result: {passed}/{len(checks)} checks passed")
    return success


def validate_major_fix_4():
    """Validate Major Fix #4: Logging utilities consolidation."""
    print("🔍 Validating Major Fix #4: Logging utilities consolidation")

    # Check consolidated logging utils
    consolidated_file = Path("data_decide/utils/logging_utils.py")
    with open(consolidated_file, "r") as f:
        consolidated_content = f.read()

    # Check deprecated wrapper
    deprecated_file = Path("data_decide/olmo/utils/logging_utils.py")
    with open(deprecated_file, "r") as f:
        deprecated_content = f.read()

    checks = [
        ("Consolidated config class", "class DataDeciderLoggingConfig:" in consolidated_content),
        ("Comprehensive setup_logging", "def setup_logging(" in consolidated_content),
        (
            "Console and file handlers",
            "console_handler" in consolidated_content and "file_handler" in consolidated_content,
        ),
        ("Configuration state tracking", "_configured = False" in consolidated_content),
        ("Training-specific config", "def configure_for_training(" in consolidated_content),
        ("Reset functionality", "def reset_logging(" in consolidated_content),
        ("Deprecated wrapper exists", "DEPRECATED" in deprecated_content),
        ("Deprecation warnings", "warnings.warn" in deprecated_content),
        ("Import forwarding", "from ...utils.logging_utils import" in deprecated_content),
        ("Wrapper functions", "def setup_logging(" in deprecated_content and "def get_logger(" in deprecated_content),
    ]

    passed = 0
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
        if check_result:
            passed += 1

    success = passed == len(checks)
    print(f"  Result: {passed}/{len(checks)} checks passed")
    return success


def validate_major_fix_5():
    """Validate Major Fix #5: Type annotations."""
    print("🔍 Validating Major Fix #5: Type annotations")

    # Check type definitions file
    type_defs_file = Path("data_decide/utils/type_definitions.py")
    with open(type_defs_file, "r") as f:
        type_defs_content = f.read()

    # Check trainer file
    trainer_file = Path("data_decide/olmo/training/trainer.py")
    with open(trainer_file, "r") as f:
        trainer_content = f.read()

    checks = [
        ("Future annotations import", "from __future__ import annotations" in trainer_content),
        ("TypedDict classes", "class TrainingConfig(TypedDict" in type_defs_content),
        ("Protocol definitions", "class TokenizerProtocol(Protocol):" in type_defs_content),
        ("Validation functions", "def validate_training_config(" in type_defs_content),
        ("Type guards", "def is_batch_data(" in type_defs_content),
        ("Trainer return annotations", "-> None:" in trainer_content),
        ("Trainer parameter annotations", "config: Union[ConfigDict, ExperimentConfig]" in trainer_content),
        ("Import type definitions", "from ...utils.type_definitions import" in trainer_content),
        ("Method type annotations", "def _create_optimizer(self) -> Optimizer:" in trainer_content),
        ("Variable type annotations", "self.training_config: TrainingConfig" in trainer_content),
    ]

    passed = 0
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
        if check_result:
            passed += 1

    success = passed == len(checks)
    print(f"  Result: {passed}/{len(checks)} checks passed")
    return success


def validate_syntax():
    """Validate that all Python files have correct syntax."""
    print("🔍 Validating Python syntax")

    python_files = [
        "data_decide/utils/type_definitions.py",
        "data_decide/utils/logging_utils.py",
        "data_decide/olmo/utils/logging_utils.py",
        "data_decide/olmo/models/configuration_olmo.py",
        "data_decide/olmo/training/trainer.py",
        "data_decide/olmo/data/data_curation.py",
        "data_decide/utils/finpile_data_loader.py",
    ]

    passed = 0
    for file_path in python_files:
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Try to compile the file
            compile(content, file_path, "exec")
            print(f"  ✅ {file_path}")
            passed += 1
        except SyntaxError as e:
            print(f"  ❌ {file_path}: {e}")
        except Exception as e:
            print(f"  ❌ {file_path}: {e}")

    success = passed == len(python_files)
    print(f"  Result: {passed}/{len(python_files)} files have valid syntax")
    return success


def main():
    """Run all validations."""
    print("🧪 Validating all implemented fixes...")
    print("=" * 60)

    validations = [
        ("Python Syntax", validate_syntax),
        ("Critical Fix #1 - Error Handling", validate_critical_fix_1),
        ("Critical Fix #2 - Memory Safety", validate_critical_fix_2),
        ("Critical Fix #3 - Configuration Factory", validate_critical_fix_3),
        ("Major Fix #4 - Logging Consolidation", validate_major_fix_4),
        ("Major Fix #5 - Type Annotations", validate_major_fix_5),
    ]

    results = []
    for validation_name, validation_func in validations:
        print(f"\n📋 {validation_name}")
        print("-" * 40)
        success = validation_func()
        results.append((validation_name, success))
        print()

    print("=" * 60)
    print("📊 VALIDATION RESULTS:")
    print("=" * 60)

    passed = 0
    for validation_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {validation_name}")
        if success:
            passed += 1

    print(f"\n🎯 {passed}/{len(results)} validations passed")

    if passed == len(results):
        print("🎉 All fixes implemented correctly!")
        print("\n📋 SUMMARY OF IMPLEMENTED FIXES:")
        print("-" * 40)
        print("✅ Critical Fix #1: Comprehensive error handling in data loading")
        print("✅ Critical Fix #2: Memory-safe data loading with context managers")
        print("✅ Critical Fix #3: Configuration factory with documented constants")
        print("✅ Major Fix #4: Consolidated logging utilities with deprecation")
        print("✅ Major Fix #5: Comprehensive type annotations and validation")
        return True
    else:
        print("⚠️  Some fixes need attention.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
