### Pipeline ETL

import logging
from pathlib import Path
from typing import List, Optional
import json
import pandas as pd

from .schema import UnifiedTextRecord
from .loaders import load_reddit_comments
from .data_transformers import reddit_to_unified

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Pipeline ETL para procesar datos de Reddit."""
    
    def __init__(
        self,
        reddit_data_dir: str = "data/historical",
        output_dir: str = "data/preprocesada"
    ):
        self.reddit_data_dir = reddit_data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_records: List[UnifiedTextRecord] = []
    
    def process_reddit(self) -> int:
        """Procesa datos de Reddit."""
        logger.info("Procesando datos de Reddit...")
        
        df = load_reddit_comments(self.reddit_data_dir)
        
        if df.empty:
            logger.warning("No se encontraron datos de Reddit")
            return 0
        
        self.all_records = reddit_to_unified(df)
        logger.info(f"Procesados {len(self.all_records)} registros")
        return len(self.all_records)
    
    def save_processed(self, filename: Optional[str] = None):
        """Guarda los registros procesados en archivo JSONL."""
        if not self.all_records:
            return
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unified_data_{timestamp}"
        
        output_path = self.output_dir / f"{filename}.jsonl"
        self._save_jsonl(output_path)
        logger.info(f"Guardados {len(self.all_records)} registros en {output_path}")
    
    def _save_jsonl(self, output_path: Path):
        """Guarda registros en formato JSONL."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in self.all_records:
                record_dict = record.model_dump()
                record_dict['created_at'] = record_dict['created_at'].isoformat()
                f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
    
    def run(self, output_filename: Optional[str] = None):
        """Ejecuta el pipeline completo."""
        self.process_reddit()
        if self.all_records:
            self.save_processed(filename=output_filename)

