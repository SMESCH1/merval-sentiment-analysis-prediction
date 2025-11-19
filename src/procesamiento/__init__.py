### Permite inicializar el módulo de procesamiento entero como piéline, evitar importaciones largas, 

from .schema import UnifiedTextRecord, DataSource
from .loaders import load_reddit_comments
from .data_transformers import reddit_to_unified
from .cleaners import clean_text, detect_language, normalize_sentiment_label
from .etl_pipeline import ETLPipeline
from .sentiment_analyzer import SentimentAnalyzer
from .sentiment_pipeline import SentimentPipeline

__all__ = [
    'UnifiedTextRecord',
    'DataSource',
    'load_reddit_comments',
    'reddit_to_unified',
    'clean_text',
    'detect_language',
    'normalize_sentiment_label',
    'ETLPipeline',
    'SentimentAnalyzer',
    'SentimentPipeline',
]

