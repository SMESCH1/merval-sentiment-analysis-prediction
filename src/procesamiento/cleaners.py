### Módulo para limpieza y normalización de texto.

import re
from typing import Optional
import logging

try:
    import langdetect
    from langdetect import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect no está disponible. La detección de idioma estará deshabilitada.")

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Limpia y normaliza texto básico.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Convertir a string y eliminar espacios al inicio y final
    cleaned = text.strip()
    
    # Normalizar espacios múltiples a uno solo
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Eliminar caracteres de control (excepto saltos de línea y tabs)
    cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
    
    # Normalizar saltos de línea múltiples
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    
    return cleaned.strip()


def detect_language(text: str) -> Optional[str]:
    """
    Detecta el idioma del texto.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Código de idioma ISO 639-1 (ej: 'es', 'en') o None si no se puede detectar
    """
    if not LANGDETECT_AVAILABLE:
        return None
    
    if not text or len(text.strip()) < 10:
        return None
    
    try:
        # langdetect requiere al menos algunos caracteres
        lang = langdetect.detect(text)
        return lang
    except LangDetectException as e:
        logger.debug(f"No se pudo detectar idioma: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error inesperado en detección de idioma: {e}")
        return None


def normalize_sentiment_label(label: str) -> Optional[str]:
    """
    Normaliza etiquetas de sentimiento a formato estándar (POS, NEG, NEU).
    
    Args:
        label: Etiqueta de sentimiento (puede estar en diferentes formatos)
        
    Returns:
        Etiqueta normalizada (POS, NEG, NEU) o None
    """
    if not label or not isinstance(label, str):
        return None
    
    label_upper = label.strip().upper()
    
    # Mapeo de variaciones comunes
    positive_variants = ['POS', 'POSITIVE', 'POSITIVO', 'POSITIVA', '1']
    negative_variants = ['NEG', 'NEGATIVE', 'NEGATIVO', 'NEGATIVA', '-1', '0']
    neutral_variants = ['NEU', 'NEUTRAL', 'NEUTRO', 'NEUTRA', '0']
    
    if label_upper in positive_variants:
        return 'POS'
    elif label_upper in negative_variants:
        return 'NEG'
    elif label_upper in neutral_variants:
        return 'NEU'
    else:
        # Si no coincide con ninguna variante conocida, retornar None
        logger.debug(f"Etiqueta de sentimiento no reconocida: {label}")
        return None

