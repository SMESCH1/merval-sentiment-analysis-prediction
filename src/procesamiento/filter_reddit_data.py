# FIltro y unifica datos de reddit, guarda solo texto y fecha


import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

def filter_and_unify_reddit_data(
    input_dir: str = "data/historical-filtered-2024-2025",
    output_path: str = "data/preprocesada/reddit_unified_2025.csv",
    start_date: str = "2025-01-05"
):
    """
    Filtra datos de Reddit desde start_date y unifica en un solo archivo.
    
    Args:
        input_dir: Directorio con archivos CSV de Reddit
        output_path: Ruta de salida para el archivo unificado
        start_date: Fecha de inicio (YYYY-MM-DD)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directorio no existe: {input_dir}")
    
    start_dt = pd.to_datetime(start_date)
    
    all_data = []
    
    # Buscar todos los archivos CSV
    csv_files = list(input_path.glob("*.csv"))
    print(f"Encontrados {len(csv_files)} archivos CSV")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            if 'created_utc' not in df.columns:
                continue
            
            # Convertir fecha
            df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
            df = df.dropna(subset=['created_utc'])
            
            # Filtrar por fecha
            df = df[df['created_utc'] >= start_dt]
            
            if df.empty:
                continue
            
            # Extraer texto y fecha
            for _, row in df.iterrows():
                # Para posts: combinar title + text
                if 'title' in df.columns and pd.notna(row.get('title')):
                    text = str(row.get('title', '')) + ' ' + str(row.get('text', ''))
                else:
                    text = str(row.get('text', ''))
                
                text = text.strip()
                if not text or text == 'nan':
                    continue
                
                all_data.append({
                    'text': text,
                    'created_at': row['created_utc']
                })
            
            print(f"  {csv_file.name}: {len(df)} registros después del {start_date}")
            
        except Exception as e:
            print(f"Error procesando {csv_file.name}: {e}")
            continue
    
    if not all_data:
        print("No se encontraron datos después de la fecha especificada")
        return
    
    # Crear DataFrame unificado
    df_unified = pd.DataFrame(all_data)
    
    # Eliminar duplicados por texto y fecha
    initial_count = len(df_unified)
    df_unified = df_unified.drop_duplicates(subset=['text', 'created_at'])
    final_count = len(df_unified)
    
    print(f"\nTotal registros: {initial_count}")
    print(f"Registros únicos: {final_count}")
    print(f"Duplicados eliminados: {initial_count - final_count}")
    
    # Ordenar por fecha
    df_unified = df_unified.sort_values('created_at').reset_index(drop=True)
    
    # Guardar
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    df_unified.to_csv(output_path, index=False)
    print(f"\nArchivo guardado: {output_path}")
    print(f"Rango de fechas: {df_unified['created_at'].min()} a {df_unified['created_at'].max()}")
    print(f"Total de días: {(df_unified['created_at'].max() - df_unified['created_at'].min()).days + 1}")


def main():
    parser = argparse.ArgumentParser(
        description='Filtra y unifica datos de Reddit desde una fecha específica'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/historical-filtered-2024-2025',
        help='Directorio con archivos CSV de Reddit'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/preprocesada/reddit_unified_2025.csv',
        help='Ruta de salida para el archivo unificado'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-01-05',
        help='Fecha de inicio (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    filter_and_unify_reddit_data(
        input_dir=args.input_dir,
        output_path=args.output,
        start_date=args.start_date
    )


if __name__ == "__main__":
    main()

