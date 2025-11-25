### Transformar la data a UnifiedTextRecord

import pandas as pd
from datetime import datetime
from typing import List, Optional
import logging
import uuid

from .schema import UnifiedTextRecord, DataSource
from .cleaners import clean_text, detect_language, normalize_sentiment_label

logger = logging.getLogger(__name__)


def reddit_to_unified(df: pd.DataFrame) -> List[UnifiedTextRecord]:
    """
    Convierte un DataFrame de comentarios de Reddit a lista de UnifiedTextRecord.
    
    Args:
        df: DataFrame con columnas de Reddit (id, text, author, subreddit, score, created_utc, etc.)
        
    Returns:
        Lista de UnifiedTextRecord validados
    """
    records = []
    
    if df.empty:
        logger.warning("DataFrame de Reddit está vacío")
        return records
    
    required_columns = ['id', 'text', 'created_utc']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en DataFrame de Reddit: {missing_columns}")
    
    for idx, row in df.iterrows():
        try:
            # Limpiar texto
            text = str(row.get('text', '')).strip()
            if not text or text == 'nan':
                continue
            
            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue
            
            # Convertir fecha
            created_utc = row.get('created_utc')
            if pd.isna(created_utc):
                logger.warning(f"Fila {idx} sin fecha, usando fecha actual")
                created_at = datetime.now()
            else:
                try:
                    if isinstance(created_utc, str):
                        created_at = pd.to_datetime(created_utc)
                    else:
                        created_at = pd.to_datetime(created_utc, unit='s', errors='coerce')
                        if pd.isna(created_at):
                            created_at = datetime.now()
                    created_at = created_at.to_pydatetime()
                except Exception as e:
                    logger.warning(f"Error al parsear fecha en fila {idx}: {e}, usando fecha actual")
                    created_at = datetime.now()
            
            # Detectar idioma
            lang = detect_language(cleaned_text)
            
            # Crear registro unificado
            record = UnifiedTextRecord(
                id=f"reddit_{row['id']}",
                source=DataSource.REDDIT,
                source_id=str(row['id']),
                text=text,  # Texto original
                created_at=created_at,
                author=str(row.get('author', '')) if pd.notna(row.get('author')) else None,
                subreddit=str(row.get('subreddit', '')) if pd.notna(row.get('subreddit')) else None,
                score=int(row.get('score', 0)) if pd.notna(row.get('score')) else None,
                cleaned_text=cleaned_text,
                text_length=len(cleaned_text),
                lang=lang,
                metadata={
                    'parent_id': str(row.get('parent_id', '')) if pd.notna(row.get('parent_id')) else None,
                    'link_id': str(row.get('link_id', '')) if pd.notna(row.get('link_id')) else None,
                    'type': str(row.get('type', 'comment')) if pd.notna(row.get('type')) else 'comment'
                }
            )
            
            records.append(record)
            
        except Exception as e:
            logger.error(f"Error al procesar fila {idx} de Reddit: {e}")
            continue
    
    logger.info(f"Convertidos {len(records)} registros de Reddit a UnifiedTextRecord")
    return records


