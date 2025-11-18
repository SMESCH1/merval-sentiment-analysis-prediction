

"""
Prepara datos de entrenamiento para LSTM combinando:
- Datos preprocesados de sentimiento (JSONL)
- Datos financieros de MERVAL (yfinance)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import yfinance as yf

def load_preprocessed_data(jsonl_path: str) -> pd.DataFrame:
    """Carga datos preprocesados desde JSONL."""
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'created_at' in data:
                    data['created_at'] = pd.to_datetime(data['created_at'])
                records.append(data)
            except Exception as e:
                print(f"Error al procesar línea: {e}")
                continue
    
    return pd.DataFrame(records)

def create_daily_sentiment_features(df_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features diarias agregadas de sentimiento.
    
    Returns:
        DataFrame con una fila por día y features de sentimiento
    """
    if len(df_sentiment) == 0:
        return pd.DataFrame()
    
    # Extraer fecha
    df_sentiment['date'] = pd.to_datetime(df_sentiment['created_at']).dt.date
    
    daily_features = []
    
    for date in df_sentiment['date'].unique():
        day_data = df_sentiment[df_sentiment['date'] == date]
        
        features = {
            'date': pd.to_datetime(date),
            'sentiment_total_records': len(day_data),
            'sentiment_reddit_count': len(day_data[day_data['source'] == 'reddit']),
            'sentiment_news_count': len(day_data[day_data['source'] == 'news']),
        }
        
        # Sentimiento predicho (si está disponible)
        if 'predicted_sentiment' in day_data.columns:
            sentiment_data = day_data[day_data['predicted_sentiment'].notna()]
            
            if len(sentiment_data) > 0:
                sentiment_counts = sentiment_data['predicted_sentiment'].value_counts()
                features['sentiment_pos_count'] = sentiment_counts.get('POS', 0)
                features['sentiment_neg_count'] = sentiment_counts.get('NEG', 0)
                features['sentiment_neu_count'] = sentiment_counts.get('NEU', 0)
                
                # Score de sentimiento (-1 a 1)
                total_sentiment = features['sentiment_pos_count'] + features['sentiment_neg_count'] + features['sentiment_neu_count']
                if total_sentiment > 0:
                    features['sentiment_score'] = (features['sentiment_pos_count'] - features['sentiment_neg_count']) / total_sentiment
                else:
                    features['sentiment_score'] = 0.0
                
                # Proporciones
                features['sentiment_pos_ratio'] = features['sentiment_pos_count'] / total_sentiment if total_sentiment > 0 else 0.0
                features['sentiment_neg_ratio'] = features['sentiment_neg_count'] / total_sentiment if total_sentiment > 0 else 0.0
                features['sentiment_neu_ratio'] = features['sentiment_neu_count'] / total_sentiment if total_sentiment > 0 else 0.0
                
                # Confianza promedio
                if 'sentiment_confidence' in sentiment_data.columns:
                    features['sentiment_avg_confidence'] = sentiment_data['sentiment_confidence'].mean()
                    features['sentiment_max_confidence'] = sentiment_data['sentiment_confidence'].max()
                    features['sentiment_min_confidence'] = sentiment_data['sentiment_confidence'].min()
                else:
                    features['sentiment_avg_confidence'] = 0.0
                    features['sentiment_max_confidence'] = 0.0
                    features['sentiment_min_confidence'] = 0.0
            else:
                # Sin datos de sentimiento para este día
                features.update({
                    'sentiment_pos_count': 0,
                    'sentiment_neg_count': 0,
                    'sentiment_neu_count': 0,
                    'sentiment_score': 0.0,
                    'sentiment_pos_ratio': 0.0,
                    'sentiment_neg_ratio': 0.0,
                    'sentiment_neu_ratio': 0.0,
                    'sentiment_avg_confidence': 0.0,
                    'sentiment_max_confidence': 0.0,
                    'sentiment_min_confidence': 0.0,
                })
        else:
            # No hay columna de sentimiento predicho - usar sentimiento manual si existe
            if 'sentiment_label' in day_data.columns:
                sentiment_data = day_data[day_data['sentiment_label'].notna()]
                if len(sentiment_data) > 0:
                    sentiment_counts = sentiment_data['sentiment_label'].value_counts()
                    features['sentiment_pos_count'] = sentiment_counts.get('POS', 0)
                    features['sentiment_neg_count'] = sentiment_counts.get('NEG', 0)
                    features['sentiment_neu_count'] = sentiment_counts.get('NEU', 0)
                    
                    total_sentiment = features['sentiment_pos_count'] + features['sentiment_neg_count'] + features['sentiment_neu_count']
                    if total_sentiment > 0:
                        features['sentiment_score'] = (features['sentiment_pos_count'] - features['sentiment_neg_count']) / total_sentiment
                    else:
                        features['sentiment_score'] = 0.0
                else:
                    features.update({
                        'sentiment_pos_count': 0,
                        'sentiment_neg_count': 0,
                        'sentiment_neu_count': 0,
                        'sentiment_score': 0.0,
                    })
            else:
                features.update({
                    'sentiment_pos_count': 0,
                    'sentiment_neg_count': 0,
                    'sentiment_neu_count': 0,
                    'sentiment_score': 0.0,
                })
            
            # Rellenar campos faltantes
            for col in ['sentiment_pos_ratio', 'sentiment_neg_ratio', 'sentiment_neu_ratio',
                       'sentiment_avg_confidence', 'sentiment_max_confidence', 'sentiment_min_confidence']:
                if col not in features:
                    features[col] = 0.0
        
        # Longitud promedio de texto
        if 'text_length' in day_data.columns:
            features['sentiment_avg_text_length'] = day_data['text_length'].mean()
        else:
            features['sentiment_avg_text_length'] = 0.0
        
        # Scores de Reddit (si aplica)
        reddit_day = day_data[day_data['source'] == 'reddit']
        if len(reddit_day) > 0 and 'score' in reddit_day.columns:
            features['reddit_avg_score'] = reddit_day['score'].mean()
            features['reddit_total_score'] = reddit_day['score'].sum()
            features['reddit_max_score'] = reddit_day['score'].max()
        else:
            features['reddit_avg_score'] = 0.0
            features['reddit_total_score'] = 0.0
            features['reddit_max_score'] = 0.0
        
        daily_features.append(features)
    
    return pd.DataFrame(daily_features).sort_values('date')


def download_financial_data(ticker: str = "^MERV", start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Descarga datos financieros de MERVAL.
    
    Args:
        ticker: Ticker a descargar (default: ^MERV)
        start_date: Fecha de inicio (YYYY-MM-DD)
        end_date: Fecha de fin (YYYY-MM-DD)
    
    Returns:
        DataFrame con retornos logarítmicos y booleano_merval
    """
    print(f"Descargando datos de {ticker}...")
    
    if start_date is None:
        start_date = "2020-01-01"
        #start_date = "2015-01-01"
    
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    if data.empty:
        raise ValueError(f"No se pudieron descargar datos para {ticker}")
    
    # Manejar si es Series o DataFrame
    if isinstance(data, pd.Series):
        precios = data
    else:
        # Si es DataFrame, puede tener MultiIndex o columnas simples
        if isinstance(data.columns, pd.MultiIndex):
            precios = data["Close"]
            # Si Close es un DataFrame (múltiples tickers), tomar la primera columna
            if isinstance(precios, pd.DataFrame):
                precios = precios.iloc[:, 0]
        else:
            precios = data["Close"]
    
    # Asegurar que precios sea una Series 1D
    if isinstance(precios, pd.DataFrame):
        precios = precios.iloc[:, 0]
    
    # Calcular retornos logarítmicos
    retornos = np.log(precios).diff().fillna(0)
    
    # Crear booleano: 1 si sube, 0 si baja (INT, no bool)
    booleano = (retornos > 0).astype(int)
    
    # Crear DataFrame usando reset_index para asegurar arrays 1D
    df = pd.DataFrame({
        'date': retornos.index.values,  # .values para obtener array numpy 1D
        'retorno_log_merval': retornos.values.flatten(),  # .flatten() para asegurar 1D
        'booleano_merval': booleano.values.flatten()  # .flatten() para asegurar 1D
    })
    
    # Convertir date a datetime si no lo es
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Descargados {len(df)} dias de datos financieros")
    
    return df

def combine_sentiment_and_financial(
    sentiment_jsonl_path: str,
    output_csv_path: str,
    merval_ticker: str = "^MERV",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Combina datos de sentimiento con datos financieros para entrenar LSTM."""
    print("Preparando datos para LSTM...")
    
    df_sentiment = load_preprocessed_data(sentiment_jsonl_path)
    if len(df_sentiment) == 0:
        raise ValueError("No se encontraron datos en el archivo JSONL")
    
    df_sentiment_daily = create_daily_sentiment_features(df_sentiment)
    if len(df_sentiment_daily) == 0:
        raise ValueError("No se pudieron crear features diarias")
    
    min_date_sentiment = df_sentiment_daily['date'].min()
    max_date_sentiment = df_sentiment_daily['date'].max()
    
    if start_date is None:
        start_date = (min_date_sentiment - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (max_date_sentiment + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    df_financial = download_financial_data(ticker=merval_ticker, start_date=start_date, end_date=end_date)
    
    df_financial['date'] = pd.to_datetime(df_financial['date']).dt.date
    df_sentiment_daily['date'] = pd.to_datetime(df_sentiment_daily['date']).dt.date
    
    df_combined = df_financial.merge(df_sentiment_daily, on='date', how='left')
    
    sentiment_cols = [col for col in df_combined.columns 
                      if col.startswith('sentiment_') or col.startswith('reddit_')]
    df_combined[sentiment_cols] = df_combined[sentiment_cols].fillna(0.0)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    
    if 'date' in df_combined.columns:
        df_combined = df_combined.drop('date', axis=1)
    
    if 'booleano_merval' in df_combined.columns:
        cols = [c for c in df_combined.columns if c != 'booleano_merval']
        cols.append('booleano_merval')
        df_combined = df_combined[cols]
    
    df_combined.to_csv(output_csv_path, index=False)
    print(f"Dataset guardado: {output_csv_path} ({len(df_combined)} filas)")
    
    return df_combined

def find_latest_processed_jsonl(data_dir: str = "data/procesada") -> Optional[str]:
    """
    Encuentra el archivo JSONL más reciente en data/procesada.
    
    Args:
        data_dir: Directorio donde buscar archivos procesados
    
    Returns:
        Ruta al archivo más reciente o None si no se encuentra
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return None
    
    # Buscar todos los archivos JSONL
    jsonl_files = list(data_path.glob("*.jsonl"))
    
    if not jsonl_files:
        return None
    
    # Ordenar por tiempo de modificación (más reciente primero)
    jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return str(jsonl_files[0])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepara datos de entrenamiento para LSTM combinando sentimiento y datos financieros'
    )
    parser.add_argument(
        '--sentiment-jsonl',
        type=str,
        default=None,
        help='Ruta al JSONL con datos procesados (default: busca el más reciente en data/procesada/)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='src/LSTM/dataset_con_sentiment.csv',
        help='Ruta de salida para el CSV combinado'
    )
    parser.add_argument(
        '--merval-ticker',
        type=str,
        default='^MERV',
        help='Ticker de MERVAL (default: ^MERV)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Fecha de inicio para datos financieros (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='Fecha de fin para datos financieros (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Si no se especifica archivo, buscar el más reciente en data/procesada
    if args.sentiment_jsonl is None:
        latest_file = find_latest_processed_jsonl()
        if latest_file is None:
            print("No se encontro archivo JSONL en data/procesada/")
            print("Ejecuta primero: python src/procesamiento/run_sentiment.py <archivo_preprocesado>")
            return
        args.sentiment_jsonl = latest_file
        print(f"Usando archivo mas reciente: {args.sentiment_jsonl}")
    
    combine_sentiment_and_financial(
        sentiment_jsonl_path=args.sentiment_jsonl,
        output_csv_path=args.output_csv,
        merval_ticker=args.merval_ticker,
        start_date=args.start_date,
        end_date=args.end_date
    )

if __name__ == "__main__":
    main()