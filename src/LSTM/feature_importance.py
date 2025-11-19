"""
Análisis de importancia de features usando diferentes métodos.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.LSTM.train_boolean_lstm import (
    TrainingConfig,
    load_dataset,
    build_sequences,
    scale_windows,
    BooleanLSTM,
    train_one_fold
)


def correlation_importance(csv_path: str):
    """Importancia basada en correlación con el target."""
    df = pd.read_csv(csv_path)
    
    if 'booleano_merval' not in df.columns:
        raise ValueError("No se encontró columna 'booleano_merval'")
    
    feature_cols = [col for col in df.columns if col != 'booleano_merval']
    correlations = df[feature_cols].corrwith(df['booleano_merval']).abs()
    correlations = correlations.sort_values(ascending=False)
    
    return correlations


def random_forest_importance(csv_path: str, n_estimators: int = 100):
    """Importancia usando Random Forest."""
    df = pd.read_csv(csv_path)
    
    feature_cols = [col for col in df.columns if col != 'booleano_merval']
    X = df[feature_cols].values
    y = df['booleano_merval'].values
    
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    
    return importances


def permutation_importance_analysis(csv_path: str, n_repeats: int = 10):
    """Importancia por permutación."""
    df = pd.read_csv(csv_path)
    
    feature_cols = [col for col in df.columns if col != 'booleano_merval']
    X = df[feature_cols].values
    y = df['booleano_merval'].values
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    perm_importance = permutation_importance(
        rf, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    
    importances = pd.Series(
        perm_importance.importances_mean,
        index=feature_cols
    )
    importances = importances.sort_values(ascending=False)
    
    return importances


def lstm_attention_analysis(csv_path: str, config: TrainingConfig):
    """Análisis de importancia usando gradientes del LSTM."""
    features, labels, feature_cols = load_dataset(csv_path)
    sequences, targets = build_sequences(features, labels, config.lookback)
    
    # Split simple
    split_idx = int(len(sequences) * 0.8)
    X_train_raw = sequences[:split_idx]
    y_train = targets[:split_idx]
    X_test_raw = sequences[split_idx:]
    y_test = targets[split_idx:]
    
    X_train, X_test, _ = scale_windows(X_train_raw, X_test_raw)
    
    # Entrenar modelo
    device = torch.device(config.device)
    model = BooleanLSTM(
        input_dim=X_train.shape[-1],
        hidden_dim=config.hidden_size,
        dropout=config.dropout
    )
    model.to(device)
    
    train_one_fold(model, X_train, y_train, config)
    model.eval()
    
    # Calcular gradientes promedio
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
    X_test_tensor.requires_grad = True
    
    output = model(X_test_tensor)
    loss = nn.BCEWithLogitsLoss()(
        output.squeeze(),
        torch.from_numpy(y_test.astype(np.float32)).to(device)
    )
    
    model.zero_grad()
    loss.backward()
    
    # Promediar gradientes absolutos por feature
    gradients = X_test_tensor.grad.abs().mean(dim=(0, 1)).cpu().numpy()
    
    importances = pd.Series(gradients, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    
    return importances


def analyze_all_methods(csv_path: str, config: TrainingConfig = None):
    """Ejecuta todos los métodos de análisis de importancia."""
    print("="*60)
    print("ANÁLISIS DE IMPORTANCIA DE FEATURES")
    print("="*60)
    
    results = {}
    
    # 1. Correlación
    print("\n1. Calculando importancia por correlación...")
    try:
        corr_imp = correlation_importance(csv_path)
        results['correlation'] = corr_imp
        print(f"   ✅ Completado: {len(corr_imp)} features")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Random Forest
    print("\n2. Calculando importancia con Random Forest...")
    try:
        rf_imp = random_forest_importance(csv_path)
        results['random_forest'] = rf_imp
        print(f"   ✅ Completado: {len(rf_imp)} features")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Permutación
    print("\n3. Calculando importancia por permutación...")
    try:
        perm_imp = permutation_importance_analysis(csv_path)
        results['permutation'] = perm_imp
        print(f"   ✅ Completado: {len(perm_imp)} features")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. LSTM Gradients (si hay config)
    if config:
        print("\n4. Calculando importancia con gradientes de LSTM...")
        try:
            lstm_imp = lstm_attention_analysis(csv_path, config)
            results['lstm_gradients'] = lstm_imp
            print(f"   ✅ Completado: {len(lstm_imp)} features")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Combinar resultados
    df_all = pd.DataFrame(results)
    df_all = df_all.fillna(0)
    
    # Normalizar cada columna
    for col in df_all.columns:
        if df_all[col].max() > 0:
            df_all[col] = df_all[col] / df_all[col].max()
    
    # Promedio de importancia
    df_all['importance_avg'] = df_all.mean(axis=1)
    df_all = df_all.sort_values('importance_avg', ascending=False)
    
    # Guardar
    output_path = Path('src/LSTM/optimization/feature_importance.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path)
    print(f"\n✅ Resultados guardados: {output_path}")
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top 15 features por método
    top_n = 15
    
    # 1. Correlación
    if 'correlation' in results:
        top_corr = results['correlation'].head(top_n)
        axes[0, 0].barh(range(len(top_corr)), top_corr.values)
        axes[0, 0].set_yticks(range(len(top_corr)))
        axes[0, 0].set_yticklabels(top_corr.index, fontsize=8)
        axes[0, 0].set_xlabel('Correlación absoluta')
        axes[0, 0].set_title('Top Features - Correlación', fontweight='bold')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. Random Forest
    if 'random_forest' in results:
        top_rf = results['random_forest'].head(top_n)
        axes[0, 1].barh(range(len(top_rf)), top_rf.values, color='orange')
        axes[0, 1].set_yticks(range(len(top_rf)))
        axes[0, 1].set_yticklabels(top_rf.index, fontsize=8)
        axes[0, 1].set_xlabel('Importancia')
        axes[0, 1].set_title('Top Features - Random Forest', fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. Permutación
    if 'permutation' in results:
        top_perm = results['permutation'].head(top_n)
        axes[1, 0].barh(range(len(top_perm)), top_perm.values, color='green')
        axes[1, 0].set_yticks(range(len(top_perm)))
        axes[1, 0].set_yticklabels(top_perm.index, fontsize=8)
        axes[1, 0].set_xlabel('Importancia por permutación')
        axes[1, 0].set_title('Top Features - Permutación', fontweight='bold')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. Promedio combinado
    top_avg = df_all['importance_avg'].head(top_n)
    axes[1, 1].barh(range(len(top_avg)), top_avg.values, color='purple')
    axes[1, 1].set_yticks(range(len(top_avg)))
    axes[1, 1].set_yticklabels(top_avg.index, fontsize=8)
    axes[1, 1].set_xlabel('Importancia promedio (normalizada)')
    axes[1, 1].set_title('Top Features - Promedio Combinado', fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('src/LSTM/optimization/feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico guardado: src/LSTM/optimization/feature_importance.png")
    
    # Mostrar top 20
    print("\n" + "="*60)
    print("TOP 20 FEATURES POR IMPORTANCIA PROMEDIO")
    print("="*60)
    print(df_all[['importance_avg']].head(20))
    
    return df_all


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analiza importancia de features')
    parser.add_argument('--csv-path', type=str, default='src/LSTM/dataset_final.csv')
    parser.add_argument('--lookback', type=int, default=25)
    parser.add_argument('--hidden-size', type=int, default=112)
    parser.add_argument('--dropout', type=float, default=0.312)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        csv_path=args.csv_path,
        lookback=args.lookback,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        device=args.device,
        epochs=10,
        batch_size=8,
        learning_rate=0.000395
    )
    
    df_importance = analyze_all_methods(args.csv_path, config)

