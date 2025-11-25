"""
Módulo para cargar datos desde diferentes fuentes.
"""
import pandas as pd
import glob
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_reddit_comments(data_dir: str = "data/historical") -> pd.DataFrame:
    """
    Carga todos los archivos CSV de comentarios de Reddit.
    Soporta tanto formato diario (reddit_data_*) como histórico (reddit_historical_*).
    Ahora acepta múltiples directorios separados por comas.
    """
    # Si hay múltiples directorios separados por comas, procesarlos todos
    if ',' in data_dir:
        dirs = [d.strip() for d in data_dir.split(',')]
        logger.info(f"Procesando múltiples directorios: {dirs}")
        dfs = []
        for dir_path in dirs:
            df = load_reddit_comments(dir_path)  # Llamada recursiva
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Consolidar todos los DataFrames de todos los directorios
        consolidated_df = pd.concat(dfs, ignore_index=True)
        initial_count = len(consolidated_df)
        consolidated_df = consolidated_df.drop_duplicates(subset=['id'], keep='first')
        
        logger.info(f"Total consolidado de {len(dirs)} directorios: {len(consolidated_df)} registros únicos (eliminados {initial_count - len(consolidated_df)} duplicados)")
        return consolidated_df
    
    # Código original para un solo directorio
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directorio {data_dir} no existe")
    
    # Buscar archivos CSV de comentarios (formato diario y histórico)
    pattern1 = str(data_path / "reddit_data_*_*_comments.csv")
    pattern2 = str(data_path / "reddit_historical_*_*_comments.csv")
    csv_files = glob.glob(pattern1) + glob.glob(pattern2)
    
    # También cargar posts
    pattern3 = str(data_path / "reddit_data_*_*_posts.csv")
    pattern4 = str(data_path / "reddit_historical_*_*_posts.csv")
    post_files = glob.glob(pattern3) + glob.glob(pattern4)
    
    if not csv_files and not post_files:
        logger.warning(f"No se encontraron archivos CSV de Reddit en {data_dir}")
        return pd.DataFrame()
    
    logger.info(f"Encontrados {len(csv_files)} archivos de comentarios y {len(post_files)} archivos de posts en {data_dir}")
    
    # Cargar comentarios
    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"Error al cargar {file_path}: {e}")
            continue
    
    # Cargar posts
    for file_path in post_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                if 'title' in df.columns and 'text' in df.columns:
                    df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
                elif 'title' in df.columns:
                    df['text'] = df['title']
                dfs.append(df)
        except Exception as e:
            logger.error(f"Error al cargar {file_path}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    # Consolidar y eliminar duplicados
    consolidated_df = pd.concat(dfs, ignore_index=True)
    initial_count = len(consolidated_df)
    consolidated_df = consolidated_df.drop_duplicates(subset=['id'], keep='first')
    
    logger.info(f"Total cargado desde {data_dir}: {len(consolidated_df)} registros únicos (eliminados {initial_count - len(consolidated_df)} duplicados)")
    
    return consolidated_df



