#!/usr/bin/env python3
"""
Script CLI para ejecutar análisis de sentimiento.
"""
import sys
import argparse
import logging
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.procesamiento.sentiment_pipeline import SentimentPipeline

def setup_logging(verbose: bool = False):
    """Configura el sistema de logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    parser = argparse.ArgumentParser(
        description='Análisis de sentimiento con pysentimiento',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Análisis básico (un archivo)
  python run_sentiment.py data/preprocesada/unified_data_20251108_121326.jsonl
  
  # Solo datos de Reddit con batch size grande (GPU recomendado)
  python run_sentiment.py data/preprocesada/unified_data_20251108_121326.jsonl --filter-source reddit --batch-size 128
  
  # Múltiples archivos
  python run_sentiment.py data/preprocesada/unified_data_*.jsonl
  
  # Especificar archivo de salida (solo con un archivo)
  python run_sentiment.py data/preprocesada/unified_data_20251108_121326.jsonl -o data/procesada/resultado.jsonl
  
  # Con batch size personalizado y GPU explícito
  python run_sentiment.py data/preprocesada/unified_data_20251108_121326.jsonl --batch-size 64 --device cuda
        """
    )
    
    parser.add_argument('input', type=str, nargs='+', help='Archivo(s) JSONL preprocesado(s) (puede especificar múltiples)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Archivo de salida (solo válido con un archivo de entrada). Con múltiples archivos, se genera automáticamente para cada uno.')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch (default: 32). Aumentar a 64-128 si tienes GPU.')
    parser.add_argument('--lang', type=str, default='es', help='Idioma del modelo (default: es)')
    parser.add_argument('--model-name', type=str, default=None, help='Nombre del modelo (opcional)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Dispositivo a usar (cuda/cpu). Si no se especifica, auto-detecta GPU.')
    parser.add_argument('--filter-source', type=str, default=None, help='Filtrar por fuente (ej: "reddit" para solo procesar datos de Reddit)')
    parser.add_argument('--no-skip', action='store_true', help='Re-analizar todos los registros (incluso si ya tienen sentimiento)')
    parser.add_argument('--use-original-text', action='store_true', help='Usar text en lugar de cleaned_text')
    parser.add_argument('--neu-to-pos-threshold', type=float, default=0.3, help='Threshold para reclasificar NEU cercanos a POS como POS (default: 0.3)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Logs detallados')
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    # Convertir a lista si es un solo archivo
    input_files = args.input if isinstance(args.input, list) else [args.input]
    
    # Validar que todos los archivos existen
    input_paths = []
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Archivo no encontrado: {input_path}")
            sys.exit(1)
        input_paths.append(input_path)
    
    # Validar output: solo puede especificarse con un archivo
    if args.output and len(input_paths) > 1:
        logger.error("--output solo puede usarse con un archivo de entrada. Con múltiples archivos, se genera automáticamente.")
        sys.exit(1)
    
    # Procesar archivos
    pipeline = SentimentPipeline(
        model_lang=args.lang, 
        model_name=args.model_name,
        device=args.device
    )
    errors = []
    
    for idx, input_path in enumerate(input_paths):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Procesando archivo {idx + 1}/{len(input_paths)}: {input_path}")
            if args.filter_source:
                logger.info(f"Filtrando por fuente: {args.filter_source}")
            logger.info(f"{'='*60}")
            
            # Determinar output_path
            if args.output and len(input_paths) == 1:
                output_path = Path(args.output)
            else:
                # Generar automáticamente
                output_path = None
            
            pipeline.run(
                input_path=input_path,
                output_path=output_path,
                batch_size=args.batch_size,
                skip_if_exists=not args.no_skip,
                use_cleaned_text=not args.use_original_text,
                filter_source=args.filter_source,
                neu_to_pos_threshold=args.neu_to_pos_threshold
            )
            logger.info(f"Archivo procesado exitosamente: {input_path}")
            
        except Exception as e:
            error_msg = f"Error procesando {input_path}: {e}"
            logger.error(error_msg, exc_info=args.verbose)
            errors.append(error_msg)
    
    # Resumen
    if errors:
        logger.error(f"\n{len(errors)} archivo(s) fallaron de {len(input_paths)}")
        for error in errors:
            logger.error(f"   - {error}")
        sys.exit(1)
    else:
        logger.info(f"\nTodos los archivos procesados exitosamente ({len(input_paths)} archivo(s))")
        sys.exit(0)

if __name__ == "__main__":
    main()

