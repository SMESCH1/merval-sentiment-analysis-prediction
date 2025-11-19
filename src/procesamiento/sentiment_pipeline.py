### Pipeline análisis de sentimiento a datos preprocesados

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import logging

from .schema import UnifiedTextRecord
from .sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class SentimentPipeline:
    """Pipeline para análisis de sentimiento."""
    
    def __init__(self, model_lang: str = "es", model_name: str = None, device: Optional[str] = None):
        """Inicializa el pipeline de analisis de sentimiento."""
        self.analyzer = SentimentAnalyzer(lang=model_lang, model_name=model_name, device=device)
    
    def load_preprocessed_data(self, input_path: Path) -> List[UnifiedTextRecord]:
        """Carga datos preprocesados desde JSONL."""
        records = []
        logger.info(f"Cargando datos desde {input_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        # Convertir created_at de string a datetime
                        if isinstance(data.get('created_at'), str):
                            data['created_at'] = datetime.fromisoformat(data['created_at'])
                        # Convertir sentiment_analyzed_at si existe
                        if 'sentiment_analyzed_at' in data and isinstance(data['sentiment_analyzed_at'], str):
                            data['sentiment_analyzed_at'] = datetime.fromisoformat(data['sentiment_analyzed_at'])
                        records.append(UnifiedTextRecord(**data))
                    except Exception as e:
                        logger.warning(f"Error al procesar línea {line_num}: {e}")
                        continue
            
            logger.info(f"Cargados {len(records)} registros")
            return records
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {input_path}")
            raise
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
    
    def analyze_records(
        self, 
        records: List[UnifiedTextRecord],
        batch_size: int = 32,
        skip_if_exists: bool = True,
        use_cleaned_text: bool = True,
        filter_source: Optional[str] = None,
        neu_to_pos_threshold: float = 0.3
    ) -> List[UnifiedTextRecord]:
        """Aplica analisis de sentimiento a los registros."""
        # Filtrar por fuente si se especifica
        if filter_source:
            original_count = len(records)
            records = [r for r in records if r.source == filter_source]
            logger.info(f"Filtrando por fuente '{filter_source}': {len(records)}/{original_count} registros")
        
        # Filtrar registros que necesitan análisis
        if skip_if_exists:
            to_analyze = [r for r in records if not r.predicted_sentiment]
            already_analyzed = [r for r in records if r.predicted_sentiment]
        else:
            to_analyze = records
            already_analyzed = []
        
        if not to_analyze:
            logger.info("No hay registros nuevos para analizar")
            return records
        
        logger.info(f"Analizando sentimiento de {len(to_analyze)} registros")
        
        # Extraer textos (usar cleaned_text si está disponible y use_cleaned_text=True)
        texts = []
        for r in to_analyze:
            if use_cleaned_text and r.cleaned_text:
                texts.append(r.cleaned_text)
            else:
                texts.append(r.text)
        
        # Analizar en batch
        results = self.analyzer.predict_batch(texts, batch_size=batch_size)
        
        # Enriquecer registros
        enriched_records = []
        analyzed_at = datetime.now()
        
        for record, result in zip(to_analyze, results):
            # Crear nuevo registro con sentimiento predicho
            record_dict = record.model_dump()
            
            # Aplicar lógica de reclasificación: NEU cercanos a POS -> POS
            predicted_label = result['label']
            probs = result.get('probs', {})
            
            if predicted_label == 'NEU' and probs:
                pos_prob = probs.get('POS', 0.0)
                if pos_prob >= neu_to_pos_threshold:
                    predicted_label = 'POS'
                    logger.debug(f"Reclasificando NEU -> POS (prob POS: {pos_prob:.3f})")
            
            record_dict['predicted_sentiment'] = predicted_label
            record_dict['sentiment_confidence'] = result['confidence']
            record_dict['sentiment_probs'] = result['probs']
            record_dict['sentiment_model'] = 'pysentimiento-robertuito'
            record_dict['sentiment_analyzed_at'] = analyzed_at
            
            try:
                enriched_records.append(UnifiedTextRecord(**record_dict))
            except Exception as e:
                logger.warning(f"Error al crear registro enriquecido para {record.id}: {e}")
                # Mantener registro original si falla
                enriched_records.append(record)
        
        # Agregar registros que no se analizaron (si skip_if_exists)
        enriched_records.extend(already_analyzed)
        
        logger.info(f"Analisis completado: {len(enriched_records)} registros enriquecidos")
        
        return enriched_records
    
    def save_enriched_data(self, records: List[UnifiedTextRecord], output_path: Path):
        """Guarda registros enriquecidos en JSONL."""
        logger.info(f"Guardando {len(records)} registros en {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                record_dict = record.model_dump()
                record_dict['created_at'] = record_dict['created_at'].isoformat()
                if record_dict.get('sentiment_analyzed_at'):
                    record_dict['sentiment_analyzed_at'] = record_dict['sentiment_analyzed_at'].isoformat()
                f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
        
        logger.info(f"Datos guardados")
    
    def run(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        batch_size: int = 32,
        skip_if_exists: bool = True,
        use_cleaned_text: bool = True,
        filter_source: Optional[str] = None,
        neu_to_pos_threshold: float = 0.3
    ):
        """Ejecuta el pipeline completo."""
        # Generar output_path si no se proporciona
        if output_path is None:
            input_stem = input_path.stem
            output_path = input_path.parent.parent / "procesada" / f"{input_stem}_with_sentiment.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cargar datos
        records = self.load_preprocessed_data(input_path)
        
        # Analizar
        enriched_records = self.analyze_records(
            records, 
            batch_size=batch_size,
            skip_if_exists=skip_if_exists,
            use_cleaned_text=use_cleaned_text,
            filter_source=filter_source,
            neu_to_pos_threshold=neu_to_pos_threshold
        )
        
        # Guardar
        self.save_enriched_data(enriched_records, output_path)
        
        logger.info("Pipeline de sentimiento completado")
        total = len(enriched_records)
        with_sentiment = sum(1 for r in enriched_records if r.predicted_sentiment)
        logger.info(f"Registros con sentimiento: {with_sentiment}/{total}")

