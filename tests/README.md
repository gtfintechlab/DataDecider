# DataDecider Test Suite

This directory contains comprehensive tests for the DataDecider project.

## Test Structure

- `conftest.py` - Shared pytest fixtures and utilities
- `test_validation_error_handling.py` - Tests for validation and error handling systems (Issues #7 & #8)
- `test_validation_system.py` - Comprehensive validation system tests
- `test_finpile_integration.py` - FinPile data integration tests
- `test_trainer_components.py` - Refactored trainer component tests (Issue #6)

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test categories:
```bash
# Unit tests only
pytest tests/ -m unit

# Integration tests only  
pytest tests/ -m integration

# FinPile-specific tests
pytest tests/ -m finpile

# Exclude slow tests
pytest tests/ -m "not slow"
```

### Run specific test files:
```bash
pytest tests/test_validation_error_handling.py
pytest tests/test_finpile_integration.py
pytest tests/test_trainer_components.py::test_trainer_initialization
```

### Generate coverage report:
```bash
pytest tests/ --cov=data_decide --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Individual component functionality
- Module imports and basic operations  
- Error handling and validation logic
- Configuration parsing and validation

### Integration Tests (`@pytest.mark.integration`)
- Component interaction testing
- End-to-end workflow validation
- File structure and architecture verification
- Cross-module compatibility

### FinPile Tests (`@pytest.mark.finpile`)
- FinPile data format compatibility
- Pre-tokenized data loading
- DataDecide methodology with FinPile data
- Memory management and safety features

### Slow Tests (`@pytest.mark.slow`)
- Performance-intensive operations
- Large dataset processing
- Training loop validation
- Memory stress testing

## Key Test Coverage Areas

### Code Quality Improvements (Issues #5-8)
- **Issue #5**: Type annotations and TypedDict validation
- **Issue #6**: Component-based trainer architecture (Single Responsibility)
- **Issue #7**: Input validation and early error detection
- **Issue #8**: Consistent error messaging and recovery

### FinPile Integration
- SimpleTapeDataset and PackedTapeDataset functionality
- Data loading with memory safety
- Model training compatibility
- DataDecide methodology adaptation

### Trainer Architecture
- Component manager functionality
- Single Responsibility Principle validation
- Configuration validation
- Training state management

### Validation Systems
- Complete experiment configuration validation
- Early validation context managers
- Dataset bounds checking
- Error message quality and actionability

## Test Data and Fixtures

### Available Fixtures
- `temp_dir`: Temporary directory for test files
- `sample_config`: Valid training configuration
- `invalid_config`: Invalid configuration for validation testing
- `sample_batch_data`: Mock batch data for model testing
- `mock_finpile_data`: Mock FinPile dataset files
- `sample_jsonl_data`: Sample JSON Lines data
- `error_reporter`: Test error reporter instance

### Mock Data Creation
- `create_jsonl_file`: Factory for creating test JSONL files
- `mock_finpile_data`: Creates realistic FinPile .bin/.idx file pairs

## Best Practices

### Writing New Tests
1. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
2. Follow the AAA pattern (Arrange, Act, Assert)
3. Include descriptive docstrings
4. Test both success and failure cases
5. Use fixtures for common test data
6. Mock external dependencies appropriately

### Test Organization
- Keep unit tests focused on individual components
- Use integration tests for component interactions
- Mark resource-intensive tests as `slow`
- Group related functionality in the same test file

### Performance Considerations
- Mock expensive operations (file I/O, model creation) in unit tests
- Use small data samples for testing algorithms
- Reserve full-scale testing for integration tests
- Clean up test artifacts with fixtures

## Coverage Goals

- **Statements**: 85%+
- **Branches**: 80%+
- **Functions**: 90%+
- **Critical paths**: 100%

## CI/CD Integration

Tests are designed to run in continuous integration environments:
- No external dependencies required for unit tests
- Mocked external services and file systems
- Configurable test timeouts
- Parallel test execution support

## Troubleshooting

### Common Issues
- **Import errors**: Ensure `PYTHONPATH` includes project root
- **Fixture not found**: Check `conftest.py` imports and fixture names
- **Slow tests**: Use `-m "not slow"` to skip performance tests
- **FinPile tests failing**: Ensure mock data fixtures are working correctly

### Debug Mode
Run tests with verbose output and no capture:
```bash
pytest tests/ -v -s --tb=long
```