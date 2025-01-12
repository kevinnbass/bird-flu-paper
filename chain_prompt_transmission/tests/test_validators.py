import pytest
from src.validators import InputValidator, ArticleSchema

def test_article_validation():
    """Test article validation with invalid article."""
    validator = InputValidator(ArticleSchema())
    article = {"id": "invalid"}
    error = validator.validate_article(article)
    assert error is not None
