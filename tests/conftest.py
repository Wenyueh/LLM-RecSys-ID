"""
Shared pytest fixtures for the testing infrastructure.

This module contains commonly used fixtures that can be imported
by any test module in the test suite.
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test use."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file within a temporary directory."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    return temp_file_path


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    config = MagicMock()
    config.batch_size = 32
    config.learning_rate = 0.001
    config.num_epochs = 10
    config.model_name = "test_model"
    config.data_path = "/test/data"
    return config


@pytest.fixture
def sample_user_data():
    """Provide sample user interaction data for testing."""
    return {
        "user_id": "user_123",
        "item_ids": ["item_1", "item_2", "item_3"],
        "ratings": [4.0, 5.0, 3.5],
        "timestamps": [1634567890, 1634567900, 1634567910],
        "categories": ["electronics", "books", "movies"]
    }


@pytest.fixture
def sample_item_data():
    """Provide sample item data for testing."""
    return {
        "item_id": "item_123",
        "title": "Test Item",
        "category": "electronics",
        "price": 29.99,
        "features": ["feature1", "feature2", "feature3"],
        "description": "A test item for unit testing"
    }


@pytest.fixture
def mock_model():
    """Provide a mock ML model for testing."""
    model = MagicMock()
    model.predict.return_value = np.array([0.8, 0.6, 0.9, 0.7])
    model.train.return_value = None
    model.save.return_value = True
    model.load.return_value = True
    return model


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 2054, 2003, 1037, 2047, 3231, 102]
    tokenizer.decode.return_value = "what is a new item"
    tokenizer.vocab_size = 30522
    return tokenizer


@pytest.fixture
def mock_dataset():
    """Provide a mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__ = Mock(return_value=1000)
    dataset.__getitem__ = Mock(return_value={
        "input_ids": [101, 2054, 2003, 102],
        "attention_mask": [1, 1, 1, 1],
        "labels": [1, 0, 1, 0]
    })
    return dataset


@pytest.fixture
def sample_sequential_data():
    """Provide sample sequential interaction data."""
    return [
        ["item_1", "item_2", "item_3"],
        ["item_4", "item_5"],
        ["item_6", "item_7", "item_8", "item_9"],
        ["item_10"]
    ]


@pytest.fixture
def mock_data_loader():
    """Provide a mock data loader for testing."""
    loader = MagicMock()
    loader.__iter__ = Mock(return_value=iter([
        {"input": "test input 1", "target": "test target 1"},
        {"input": "test input 2", "target": "test target 2"}
    ]))
    loader.__len__ = Mock(return_value=2)
    return loader


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = MagicMock()
    logger.info = Mock()
    logger.error = Mock()
    logger.warning = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def mock_transformers_model():
    """Provide a mock transformers model for testing."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.vocab_size = 30522
    model.config.hidden_size = 768
    model.forward.return_value = MagicMock(
        logits=np.random.randn(2, 10, 30522),
        loss=0.5
    )
    return model


@pytest.fixture
def sample_generation_data():
    """Provide sample data for ID generation testing."""
    return {
        "items": {
            "item_1": {"title": "Product A", "category": "electronics"},
            "item_2": {"title": "Product B", "category": "books"},
            "item_3": {"title": "Product C", "category": "movies"}
        },
        "users": {
            "user_1": {"age": 25, "gender": "M", "location": "NY"},
            "user_2": {"age": 30, "gender": "F", "location": "CA"}
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically setup test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def mock_file_operations():
    """Provide mocks for file I/O operations."""
    mocks = MagicMock()
    mocks.open = Mock()
    mocks.read = Mock(return_value="test file content")
    mocks.write = Mock(return_value=True)
    mocks.exists = Mock(return_value=True)
    return mocks


@pytest.fixture(params=["beauty", "sports", "yelp"])
def dataset_name(request):
    """Parametrized fixture for different dataset names."""
    return request.param


@pytest.fixture(params=[10, 20, 50])
def batch_size(request):
    """Parametrized fixture for different batch sizes."""
    return request.param