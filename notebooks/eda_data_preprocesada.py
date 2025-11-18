#!/usr/bin/env python3
"""
EDA (Exploratory Data Analysis) para archivos JSONL de datos preprocesados.
Analiza la cobertura temporal y identifica días faltantes.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import argparse

def load_jsonl(jsonl_path: str) -> pd.DataFrame:
    """Carga datos desde JSONL."""
    print(f"📂 Cargando datos desde: {jsonl_path}")
    records = []
    errors = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if 'created_at' in data:
                    data['created_at'] = pd.to_datetime(data['created_at'])
                records.append(data)
            except Exception as e:
                errors += 1
                if errors <= 5:  # Mostrar solo primeros 5 errores
                    print(f"⚠️  Error en línea {line_num}: {e}")
                continue
    
    if errors > 0:
        print(f"⚠️  Total de errores al cargar: {errors}")
    
    df = pd.DataFrame(records)
    print(f"✅ Cargados {len(df)} registros")
    return df

def analyze_temporal_coverage(df: pd.DataFrame) -> dict:
    """Analiza la cobertura temporal del dataset."""
    print("\n" + "="*60)
    print("ANÁLISIS DE COBERTURA TEMPORAL")
    print("="*60)
    
    # Extraer fecha (solo día, sin hora)
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    
    # Estadísticas básicas
    min_date = df['date'].min()
    max_date = df['date'].max()
    total_days = (max_date - min_date).days + 1
    
    # Días únicos con datos
    unique_days = df['date'].nunique()
    
    # Crear rango completo de fechas
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    date_range_dates = [d.date() for d in date_range]
    
    # Días con datos
    days_with_data = set(df['date'].unique())
    
    # Días faltantes
    missing_days = set(date_range_dates) - days_with_data
    
    # Conteo por día
    daily_counts = df.groupby('date').size()
    
    # Estadísticas por fuente
    if 'source' in df.columns:
        source_stats = df.groupby(['date', 'source']).size().unstack(fill_value=0)
    else:
        source_stats = None
    
    results = {
        'min_date': min_date,
        'max_date': max_date,
        'total_days_in_range': total_days,
        'days_with_data': unique_days,
        'days_missing': len(missing_days),
        'coverage_percentage': (unique_days / total_days * 100) if total_days > 0 else 0,
        'missing_days': sorted(list(missing_days)),
        'daily_counts': daily_counts,
        'source_stats': source_stats,
        'total_records': len(df)
    }
    
    return results

def print_summary(stats: dict):
    """Imprime resumen del análisis."""
    print(f"\n📅 Rango de fechas:")
    print(f"   Desde: {stats['min_date']}")
    print(f"   Hasta: {stats['max_date']}")
    print(f"   Total de días en rango: {stats['total_days_in_range']}")
    
    print(f"\n📊 Cobertura:")
    print(f"   Días con datos: {stats['days_with_data']}")
    print(f"   Días faltantes: {stats['days_missing']}")
    print(f"   Cobertura: {stats['coverage_percentage']:.2f}%")
    print(f"   Total de registros: {stats['total_records']:,}")
    
    if stats['daily_counts'].shape[0] > 0:
        print(f"\n📈 Estadísticas diarias:")
        print(f"   Promedio de registros por día: {stats['daily_counts'].mean():.1f}")
        print(f"   Mediana de registros por día: {stats['daily_counts'].median():.1f}")
        print(f"   Mínimo de registros en un día: {stats['daily_counts'].min()}")
        print(f"   Máximo de registros en un día: {stats['daily_counts'].max()}")
        print(f"   Desviación estándar: {stats['daily_counts'].std():.1f}")
    
    # Mostrar días faltantes (solo primeros 20)
    if stats['missing_days']:
        print(f"\n⚠️  Días faltantes (mostrando primeros 20):")
        for day in stats['missing_days'][:20]:
            print(f"   - {day}")
        if len(stats['missing_days']) > 20:
            print(f"   ... y {len(stats['missing_days']) - 20} días más")
    
    # Estadísticas por fuente
    if stats['source_stats'] is not None:
        print(f"\n📊 Registros por fuente:")
        for source in stats['source_stats'].columns:
            total = stats['source_stats'][source].sum()
            print(f"   {source}: {total:,} registros")

def plot_temporal_coverage(stats: dict, output_dir: Optional[Path] = None):
    """Genera gráficos de cobertura temporal."""
    print("\n📊 Generando gráficos...")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Cobertura Temporal', fontsize=16, fontweight='bold')
    
    # 1. Registros por día
    ax1 = axes[0, 0]
    stats['daily_counts'].plot(ax=ax1, kind='line', color='#3498db', linewidth=1.5)
    ax1.set_title('Registros por Día', fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Número de Registros')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Histograma de registros por día
    ax2 = axes[0, 1]
    stats['daily_counts'].hist(bins=50, ax=ax2, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax2.set_title('Distribución de Registros por Día', fontweight='bold')
    ax2.set_xlabel('Número de Registros')
    ax2.set_ylabel('Frecuencia (Días)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cobertura temporal (heatmap por mes)
    ax3 = axes[1, 0]
    daily_counts_df = stats['daily_counts'].reset_index()
    daily_counts_df['year'] = pd.to_datetime(daily_counts_df['date']).dt.year
    daily_counts_df['month'] = pd.to_datetime(daily_counts_df['date']).dt.month
    daily_counts_df['day'] = pd.to_datetime(daily_counts_df['date']).dt.day
    
    # Crear pivot table para heatmap
    pivot = daily_counts_df.pivot_table(
        values=0,  # El conteo está en la columna 0
        index='day',
        columns=['year', 'month'],
        fill_value=0
    )
    
    # Simplificar: mostrar solo últimos 12 meses
    if len(pivot.columns) > 12:
        pivot = pivot.iloc[:, -12:]
    
    sns.heatmap(pivot, ax=ax3, cmap='YlOrRd', cbar_kws={'label': 'Registros'}, 
                fmt='.0f', linewidths=0.5)
    ax3.set_title('Heatmap de Registros (Últimos 12 Meses)', fontweight='bold')
    ax3.set_xlabel('Mes')
    ax3.set_ylabel('Día del Mes')
    
    # 4. Registros por fuente (si existe)
    ax4 = axes[1, 1]
    if stats['source_stats'] is not None:
        source_totals = stats['source_stats'].sum()
        source_totals.plot(ax=ax4, kind='bar', color=['#e74c3c', '#3498db', '#f39c12'], 
                          edgecolor='black', alpha=0.7)
        ax4.set_title('Total de Registros por Fuente', fontweight='bold')
        ax4.set_xlabel('Fuente')
        ax4.set_ylabel('Total de Registros')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No hay información de fuente disponible', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Registros por Fuente', fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar gráfico
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'eda_temporal_coverage.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {output_file}")
    else:
        plt.savefig('eda_temporal_coverage.png', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: eda_temporal_coverage.png")
    
    plt.close()

def export_missing_days(stats: dict, output_file: str = 'missing_days.csv'):
    """Exporta días faltantes a CSV."""
    if stats['missing_days']:
        missing_df = pd.DataFrame({
            'date': stats['missing_days'],
            'day_of_week': [pd.to_datetime(d).day_name() for d in stats['missing_days']]
        })
        missing_df.to_csv(output_file, index=False)
        print(f"✅ Días faltantes exportados a: {output_file}")
    else:
        print("✅ No hay días faltantes para exportar")

def main():
    parser = argparse.ArgumentParser(
        description='EDA de archivo JSONL: analiza cobertura temporal y días faltantes'
    )
    parser.add_argument(
        'jsonl_path',
        type=str,
        help='Ruta al archivo JSONL a analizar'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directorio para guardar gráficos (default: directorio actual)'
    )
    parser.add_argument(
        '--export-missing',
        action='store_true',
        help='Exportar días faltantes a CSV'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='No generar gráficos'
    )
    
    args = parser.parse_args()
    
    # Cargar datos
    df = load_jsonl(args.jsonl_path)
    
    if len(df) == 0:
        print("❌ No se encontraron datos en el archivo")
        return
    
    # Analizar cobertura temporal
    stats = analyze_temporal_coverage(df)
    
    # Imprimir resumen
    print_summary(stats)
    
    # Generar gráficos
    if not args.no_plots:
        output_dir = Path(args.output_dir) if args.output_dir else None
        plot_temporal_coverage(stats, output_dir)
    
    # Exportar días faltantes
    if args.export_missing:
        export_missing_days(stats)
    
    print("\n" + "="*60)
    print("✅ Análisis completado")
    print("="*60)

if __name__ == "__main__":
    main()