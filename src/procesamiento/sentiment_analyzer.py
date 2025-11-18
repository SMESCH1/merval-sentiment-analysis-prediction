### Módulo para análisis de sentimiento usando pysentimiento

from pysentimiento import create_analyzer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analizador de sentimiento usando pysentimiento."""
    
    def __init__(self, lang: str = "es", model_name: str = None, device: Optional[str] = None):
        """Inicializa el analizador de sentimiento."""
        self.lang = lang
        
        # Detectar GPU automáticamente si no se especifica
        if device is None:
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if torch.cuda.is_available():
                    logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            except ImportError:
                self.device = 'cpu'
                logger.debug("PyTorch no disponible, usando CPU")
        else:
            self.device = device
        
        logger.info(f"Inicializando analizador de sentimiento para idioma: {lang} en {self.device}")
        
        # pysentimiento usa transformers por debajo, que detecta GPU automáticamente
        # Si necesitas forzar device, puedes pasar device_id en algunos casos
        self.analyzer = create_analyzer(task="sentiment", lang=lang, model_name=model_name)
        logger.info(f"Analizador de sentimiento inicializado en {self.device}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predice el sentimiento de un texto."""
        if not text or not text.strip():
            return {
                'label': None,
                'probs': None,
                'confidence': None
            }
        
        try:
            result = self.analyzer.predict(text)
            
            # Normalizar a formato POS/NEG/NEU
            label_map = {
                'POS': 'POS',
                'NEG': 'NEG',
                'NEU': 'NEU',
                'positive': 'POS',
                'negative': 'NEG',
                'neutral': 'NEU',
                'POSITIVE': 'POS',
                'NEGATIVE': 'NEG',
                'NEUTRAL': 'NEU'
            }
            
            label = label_map.get(result.output, result.output.upper() if result.output else None)
            
            # Extraer probabilidades
            probs = None
            confidence = None
            if hasattr(result, 'probas') and result.probas:
                probs = {
                    'POS': result.probas.get('POS', result.probas.get('positive', 0.0)),
                    'NEG': result.probas.get('NEG', result.probas.get('negative', 0.0)),
                    'NEU': result.probas.get('NEU', result.probas.get('neutral', 0.0))
                }
                confidence = max(probs.values())
            
            return {
                'label': label,
                'probs': probs,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Error al predecir sentimiento: {e}")
            return {
                'label': None,
                'probs': None,
                'confidence': None
            }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predice sentimiento para multiples textos en batch."""
        if not texts:
            return []
        
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Procesando {len(texts)} textos en {total_batches} batches (batch_size={batch_size})")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            processed = i + len(batch)
            
            try:
                # Mostrar progreso siempre (no solo en debug)
                percentage = (processed / len(texts) * 100) if len(texts) > 0 else 0
                logger.info(f"Batch {batch_num}/{total_batches} - Procesando {len(batch)} textos ({processed}/{len(texts)} registros, {percentage:.1f}%)")
                batch_results = self.analyzer.predict(batch)
                
                for result in batch_results:
                    label_map = {
                        'POS': 'POS', 'NEG': 'NEG', 'NEU': 'NEU',
                        'positive': 'POS', 'negative': 'NEG', 'neutral': 'NEU',
                        'POSITIVE': 'POS', 'NEGATIVE': 'NEG', 'NEUTRAL': 'NEU'
                    }
                    
                    label = label_map.get(result.output, result.output.upper() if result.output else None)
                    
                    probs = None
                    confidence = None
                    if hasattr(result, 'probas') and result.probas:
                        probs = {
                            'POS': result.probas.get('POS', result.probas.get('positive', 0.0)),
                            'NEG': result.probas.get('NEG', result.probas.get('negative', 0.0)),
                            'NEU': result.probas.get('NEU', result.probas.get('neutral', 0.0))
                        }
                        confidence = max(probs.values())
                    
                    results.append({
                        'label': label,
                        'probs': probs,
                        'confidence': confidence
                    })
            except Exception as e:
                logger.error(f"Error en batch {batch_num}: {e}")
                # Agregar resultados None para mantener consistencia
                results.extend([{
                    'label': None,
                    'probs': None,
                    'confidence': None
                }] * len(batch))
        
        return results

