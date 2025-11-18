### Esquema para los datos 

import pandas as pd
import glob
import json
from datetime import datetime
from typing import Optional, List, Any, Dict
from enum import Enum
from pydantic import BaseModel, Field, ValidationError
import langdetect



class DataSource(str, Enum):
    REDDIT = "reddit"
    NEWS = "news"
    FINANCIAL = "financial"
    EXTERNAL = "external"


class UnifiedTextRecord(BaseModel):
    """Normalizador de la estructura para las distintas fuentes de datos"""
    
    id: str
    source: DataSource
    source_id: str
    text: str
    created_at: datetime

    # verificar si estos campos a continuación valen la pena
    title: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    subreddit: Optional[str] = None
    medium: Optional[str] = None
    
    score: Optional[int] = None
    num_comments: Optional[int] = None

    sentiment_label: Optional[str] = Field(None, pattern="^(POS|NEG|NEU)$")
    tags: Optional[List[str]] = None

    # campos procesados (se llenan en ETL)?
    cleaned_text: Optional[str] = None
    text_length: Optional[int] = None
    keyword_count: Optional[int] = None
    lang: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    
    # Sentimiento predicho por modelo
    predicted_sentiment: Optional[str] = Field(None, pattern="^(POS|NEG|NEU)$")
    sentiment_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    sentiment_probs: Optional[Dict[str, float]] = None
    sentiment_model: Optional[str] = None
    sentiment_analyzed_at: Optional[datetime] = None





