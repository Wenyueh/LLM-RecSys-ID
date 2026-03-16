"""
Infrastructure validation tests.

These tests verify that the testing infrastructure is properly set up
and that basic functionality works as expected.
"""
import pytest
import numpy as np
from pathlib import Path


class TestInfrastructure:
    """Test class to validate testing infrastructure setup."""

    def test_pytest_working(self):
        """Verify that pytest is working correctly."""
        assert True

    def test_numpy_import(self):
        """Verify that numpy can be imported and used."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.mean() == 3.0

    @pytest.mark.unit
    def test_custom_marker_unit(self):
        """Test the custom 'unit' marker."""
        assert 1 + 1 == 2

    @pytest.mark.integration
    def test_custom_marker_integration(self):
        """Test the custom 'integration' marker."""
        assert "integration" in "integration test"

    @pytest.mark.slow
    def test_custom_marker_slow(self):
        """Test the custom 'slow' marker."""
        import time
        time.sleep(0.01)  # Simulate slow operation
        assert True

    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture works."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_temp_file_fixture(self, temp_file):
        """Test that temp_file fixture works."""
        assert isinstance(temp_file, Path)
        assert temp_file.exists()
        assert temp_file.is_file()
        assert temp_file.read_text() == "test content"

    def test_mock_config_fixture(self, mock_config):
        """Test that mock_config fixture provides expected attributes."""
        assert mock_config.batch_size == 32
        assert mock_config.learning_rate == 0.001
        assert mock_config.num_epochs == 10
        assert mock_config.model_name == "test_model"

    def test_sample_user_data_fixture(self, sample_user_data):
        """Test that sample_user_data fixture provides expected data."""
        assert sample_user_data["user_id"] == "user_123"
        assert len(sample_user_data["item_ids"]) == 3
        assert len(sample_user_data["ratings"]) == 3
        assert all(isinstance(r, float) for r in sample_user_data["ratings"])

    def test_mock_model_fixture(self, mock_model):
        """Test that mock_model fixture works."""
        predictions = mock_model.predict()
        assert len(predictions) == 4
        assert all(isinstance(p, (int, float)) for p in predictions)
        
        # Test that methods can be called
        mock_model.train()
        mock_model.save()
        mock_model.load()

    def test_parametrized_dataset_names(self, dataset_name):
        """Test parametrized dataset names fixture."""
        assert dataset_name in ["beauty", "sports", "yelp"]

    def test_parametrized_batch_sizes(self, batch_size):
        """Test parametrized batch sizes fixture."""
        assert batch_size in [10, 20, 50]
        assert isinstance(batch_size, int)

    def test_environment_setup(self):
        """Test that test environment is properly set up."""
        import os
        assert os.environ.get("TESTING") == "true"
        assert os.environ.get("LOG_LEVEL") == "DEBUG"


class TestMockObjects:
    """Test mock objects provided by fixtures."""

    def test_mock_tokenizer(self, mock_tokenizer):
        """Test mock tokenizer functionality."""
        encoded = mock_tokenizer.encode("test text")
        assert isinstance(encoded, list)
        assert len(encoded) == 7
        
        decoded = mock_tokenizer.decode(encoded)
        assert isinstance(decoded, str)
        assert decoded == "what is a new item"

    def test_mock_dataset(self, mock_dataset):
        """Test mock dataset functionality."""
        assert len(mock_dataset) == 1000
        
        item = mock_dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_mock_data_loader(self, mock_data_loader):
        """Test mock data loader functionality."""
        assert len(mock_data_loader) == 2
        
        items = list(mock_data_loader)
        assert len(items) == 2
        assert all("input" in item for item in items)
        assert all("target" in item for item in items)


def test_simple_arithmetic():
    """Simple test to verify basic pytest functionality."""
    assert 2 + 2 == 4
    assert 10 / 2 == 5.0
    assert 3 * 4 == 12


def test_string_operations():
    """Test string operations."""
    test_string = "Hello, World!"
    assert "Hello" in test_string
    assert test_string.lower() == "hello, world!"
    assert len(test_string) == 13


def test_list_operations():
    """Test list operations."""
    test_list = [1, 2, 3, 4, 5]
    assert len(test_list) == 5
    assert sum(test_list) == 15
    assert max(test_list) == 5
    assert min(test_list) == 1