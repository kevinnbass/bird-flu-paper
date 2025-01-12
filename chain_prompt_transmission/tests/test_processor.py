import pytest
from src.processor import ArticleProcessor

def test_processor_initialization():
    """Test ArticleProcessor initialization."""
    with pytest.raises(FileNotFoundError):
        ArticleProcessor(
            api_client=None,
            validator=None,
            config_path="nonexistent.yaml",
            prompts_path="nonexistent.yaml"
        )
