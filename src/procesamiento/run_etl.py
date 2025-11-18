### Run ETL script
# Este script se encarga de procesar los datos de Reddit y Noticias y guardarlos en un formato unificado

import sys
import argparse     
import logging
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.procesamiento.etl_pipeline import ETLPipeline


def setup_logging(verbose: bool = False):
    """
    Configura el sistema de logging.
    
    Args:
        verbose: Si mostrar logs detallados
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Función principal del script CLI."""
    parser = argparse.ArgumentParser(description='Pipeline ETL para procesar datos de Reddit')
    
    parser.add_argument(
        '--reddit-dir',
        type=str,
        default='data/historical',
        help='Directorio con datos de Reddit (default: data/historical)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/preprocesada',
        help='Directorio de salida (default: data/preprocesada)'
    )
    
    parser.add_argument(
        '--output-filename',
        type=str,
        default=None,
        help='Nombre del archivo de salida (sin extensión). Si no se especifica, se genera automáticamente.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar logs detallados'
    )
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        pipeline = ETLPipeline(
            reddit_data_dir=args.reddit_dir,
            output_dir=args.output_dir
        )
        pipeline.run(output_filename=args.output_filename)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()

