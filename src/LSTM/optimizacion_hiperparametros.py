"""
Optimizador de hiperparámetros para LSTM usando Optuna.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import optuna
from optuna.trial import TrialState

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.LSTM.train_boolean_lstm import (
    TrainingConfig, 
    walk_forward_training, 
    load_dataset,
    summarize_metrics
)


def objective(trial, csv_path: str, n_trials_per_fold: int = 3):
    """
    Función objetivo para Optuna.
    
    Args:
        trial: Trial de Optuna
        csv_path: Ruta al CSV con datos
        n_trials_per_fold: Número de folds a evaluar por trial (para acelerar)
    """
    # Sugerir hiperparámetros
    lookback = trial.suggest_int('lookback', 5, 30, step=5)
    hidden_size = trial.suggest_int('hidden_size', 16, 128, step=16)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 50, step=10)
    
    # Parámetros de walk-forward (más conservadores)
    initial_train_size = trial.suggest_int('initial_train_size', 20, 100, step=20)
    test_window = trial.suggest_int('test_window', 5, 20, step=5)
    
    # Crear configuración
    config = TrainingConfig(
        csv_path=csv_path,
        lookback=lookback,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout=dropout,
        initial_train_size=initial_train_size,
        test_window=test_window,
        device='cuda',  # Asumir GPU disponible
        save_results=False,  # No guardar en cada trial
        save_model=False
    )
    
    # Cargar datos
    features, labels, feature_cols = load_dataset(csv_path)
    
    # Entrenar con walk-forward (solo algunos folds para acelerar)
    try:
        metrics, _, _ = walk_forward_training(
            features, labels, config, return_predictions=False
        )
        
        # Usar solo los primeros n_trials_per_fold folds para evaluación rápida
        if len(metrics) > n_trials_per_fold:
            metrics = metrics[:n_trials_per_fold]
        
        # Calcular métrica objetivo (F1 promedio)
        summary = summarize_metrics(metrics)
        f1_score = summary.get('f1', 0.0)
        
        return f1_score
        
    except Exception as e:
        print(f"Error en trial {trial.number}: {e}")
        return float('-inf')


def optimize_hyperparameters(
    csv_path: str,
    n_trials: int = 50,
    study_name: str = "lstm_optimization",
    output_dir: str = "src/LSTM/optimization"
):
    """
    Optimiza hiperparámetros usando Optuna.
    
    Args:
        csv_path: Ruta al CSV con datos de entrenamiento
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio
        output_dir: Directorio para guardar resultados
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f'sqlite:///{output_path}/{study_name}.db',
        load_if_exists=True
    )
    
    print(f"Iniciando optimización con {n_trials} trials...")
    print(f"CSV: {csv_path}")
    print(f"Resultados: {output_path}")
    
    # Ejecutar optimización
    study.optimize(
        lambda trial: objective(trial, csv_path, n_trials_per_fold=3),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Resultados
    print("\n" + "="*60)
    print("OPTIMIZACIÓN COMPLETADA")
    print("="*60)
    
    print(f"\nMejor trial:")
    best_trial = study.best_trial
    print(f"  F1 Score: {best_trial.value:.4f}")
    print(f"  Parámetros:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Guardar resultados
    results = {
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE])
    }
    
    results_path = output_path / f"{study_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados guardados: {results_path}")
    
    # Guardar gráficos de optimización
    try:
        import plotly
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(str(output_path / f"{study_name}_optimization_history.html"))
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(str(output_path / f"{study_name}_param_importances.html"))
        
        print(f"Gráficos guardados en {output_path}")
    except ImportError:
        print("plotly no disponible, omitiendo gráficos")
    
    return study


def main():
    parser = argparse.ArgumentParser(
        description='Optimiza hiperparámetros del modelo LSTM'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default='src/LSTM/dataset_final.csv',
        help='Ruta al CSV con datos de entrenamiento'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Número de trials a ejecutar (default: 50)'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='lstm_optimization',
        help='Nombre del estudio de optimización'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='src/LSTM/optimization',
        help='Directorio para guardar resultados'
    )
    
    args = parser.parse_args()
    
    optimize_hyperparameters(
        csv_path=args.csv_path,
        n_trials=args.n_trials,
        study_name=args.study_name,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

