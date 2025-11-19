"""
Reclasifica sentimientos en datos ya procesados.
Reclasifica NEU cercanos a POS como POS basado en probabilidades.
"""

import json
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reclassify_neu_to_pos(
    input_path: str,
    output_path: str,
    threshold: float = 0.3
):
    """
    Reclasifica registros NEU con probabilidad POS >= threshold como POS.
    
    Args:
        input_path: Ruta al JSONL con sentimientos
        output_path: Ruta de salida
        threshold: Threshold de probabilidad POS para reclasificar (default: 0.3)
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    records_processed = 0
    records_reclassified = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                record = json.loads(line.strip())
                records_processed += 1
                
                # Solo procesar si tiene sentimiento y probabilidades
                if record.get('predicted_sentiment') == 'NEU' and record.get('sentiment_probs'):
                    probs = record['sentiment_probs']
                    pos_prob = probs.get('POS', 0.0)
                    
                    if pos_prob >= threshold:
                        record['predicted_sentiment'] = 'POS'
                        record['sentiment_confidence'] = pos_prob  # Actualizar confianza
                        records_reclassified += 1
                        logger.debug(f"Reclasificado: prob POS={pos_prob:.3f}")
                
                # Guardar registro (modificado o no)
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                
            except Exception as e:
                logger.error(f"Error procesando línea: {e}")
                continue
    
    logger.info(f"Procesados {records_processed} registros")
    logger.info(f"Reclasificados {records_reclassified} registros de NEU a POS (threshold={threshold})")
    logger.info(f"Archivo guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Reclasifica NEU cercanos a POS como POS basado en probabilidades'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Archivo JSONL de entrada con sentimientos'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Archivo de salida (default: input_reclassified.jsonl)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Threshold de probabilidad POS para reclasificar NEU (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_reclassified.jsonl")
    
    reclassify_neu_to_pos(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()

