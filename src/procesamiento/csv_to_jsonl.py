"""
Convierte el CSV unificado de Reddit a JSONL para el pipeline de sentimiento.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import argparse

def csv_to_jsonl(csv_path: str, output_path: str):
    """
    Convierte CSV unificado a JSONL compatible con el pipeline de sentimiento.
    """
    df = pd.read_csv(csv_path)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    for _, row in df.iterrows():
        record = {
            'id': f"reddit_{row.name}",
            'source': 'reddit',
            'source_id': str(row.name),
            'text': str(row['text']),
            'created_at': row['created_at'].isoformat(),
            'cleaned_text': None,
            'text_length': len(str(row['text'])),
        }
        records.append(record)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Convertidos {len(records)} registros a JSONL")
    print(f"Archivo guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convierte CSV a JSONL')
    parser.add_argument(
        '--csv',
        type=str,
        default='data/preprocesada/reddit_unified_2025.csv',
        help='Ruta al CSV de entrada'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/preprocesada/reddit_unified_2025.jsonl',
        help='Ruta de salida JSONL'
    )
    
    args = parser.parse_args()
    csv_to_jsonl(args.csv, args.output)


if __name__ == "__main__":
    main()

