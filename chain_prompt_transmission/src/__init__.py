"""Bird flu analysis processing package."""
from .api import APIClient
from .processor import ArticleProcessor
from .validators import InputValidator, ArticleSchema

__version__ = "1.0.0"
__all__ = ['APIClient', 'ArticleProcessor', 'InputValidator', 'ArticleSchema']
