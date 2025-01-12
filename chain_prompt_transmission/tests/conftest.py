"""PyTest configuration and fixtures."""
import pytest
from pathlib import Path

@pytest.fixture
def sample_article():
    """Provide a sample article for testing."""
    return {
        "id": "475",
        "fulltext": "Sample article text for testing."
    }

@pytest.fixture
def config_path():
    """Provide path to test configuration."""
    return Path("config/config.yaml")
