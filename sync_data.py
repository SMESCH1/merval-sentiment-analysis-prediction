#!/usr/bin/env python3
"""
Script de sincronización de datos entre local y GitHub
Combina datos locales con datos de GitHub Actions
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

def sync_reddit_data():
    """Sincronizar datos de Reddit entre local y GitHub."""
    
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Sincronizando datos de Reddit...")
    
    # Buscar archivos de datos
    json_files = list(data_dir.glob('reddit_data_*.json'))
    csv_files = list(data_dir.glob('reddit_data_*.csv'))
    
    print(f"📊 Encontrados {len(json_files)} archivos JSON")
    print(f"📊 Encontrados {len(csv_files)} archivos CSV")
    
    # Crear resumen de datos
    summary = {
        'last_sync': datetime.now().isoformat(),
        'total_files': len(json_files) + len(csv_files),
        'json_files': len(json_files),
        'csv_files': len(csv_files),
        'latest_data': None
    }
    
    if json_files:
        # Encontrar el archivo más reciente
        latest_file = max(json_files, key=os.path.getctime)
        summary['latest_data'] = latest_file.name
        
        # Cargar y mostrar resumen del archivo más reciente
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_posts = sum(subreddit_data.get('total_posts', 0) for subreddit_data in data.values())
            total_comments = sum(subreddit_data.get('total_comments', 0) for subreddit_data in data.values())
            
            print(f"📈 Último scraping: {latest_file.name}")
            print(f"📝 Total posts: {total_posts}")
            print(f"💬 Total comments: {total_comments}")
            
        except Exception as e:
            print(f"⚠️ Error leyendo {latest_file}: {e}")
    
    # Guardar resumen
    summary_file = data_dir / 'sync_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Sincronización completada")
    print(f"📋 Resumen guardado en: {summary_file}")

def show_data_status():
    """Mostrar estado actual de los datos."""
    
    data_dir = Path('data/raw')
    
    if not data_dir.exists():
        print("❌ No hay datos disponibles")
        return
    
    print("📊 Estado de los datos:")
    print("=" * 40)
    
    # Archivos JSON
    json_files = list(data_dir.glob('reddit_data_*.json'))
    if json_files:
        print(f"📄 Archivos JSON: {len(json_files)}")
        for file in sorted(json_files)[-3:]:  # Mostrar los 3 más recientes
            print(f"  - {file.name}")
    
    # Archivos CSV
    csv_files = list(data_dir.glob('reddit_data_*.csv'))
    if csv_files:
        print(f"📊 Archivos CSV: {len(csv_files)}")
    
    # Logs
    log_files = list(Path('logs').glob('*.log'))
    if log_files:
        print(f"📋 Archivos de log: {len(log_files)}")

def main():
    """Función principal."""
    print("🔄 Sincronizador de Datos Reddit")
    print("=" * 40)
    
    while True:
        print("\nOpciones:")
        print("1. Sincronizar datos")
        print("2. Mostrar estado de datos")
        print("3. Salir")
        
        choice = input("\nElige una opción (1-3): ").strip()
        
        if choice == '1':
            sync_reddit_data()
        elif choice == '2':
            show_data_status()
        elif choice == '3':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")

if __name__ == "__main__":
    main()
