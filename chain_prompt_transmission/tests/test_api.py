import pytest
from src.api import APIClient

def test_api_client_initialization():
    """Test APIClient initialization with invalid key."""
    with pytest.raises(ValueError):
        APIClient(api_key="", config_path="config/config.yaml")
