### LSTM en PyTorch
# Predicción de booleano_merval con validación walk-forward.

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Agregar el directorio raíz al path para importar módulos del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para guardar archivos
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    csv_path: Optional[str] = None
    jsonl_path: Optional[str] = None
    merval_ticker: str = "^MERV"
    lookback: int = 20
    hidden_size: int = 64
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    clip_value: float = 1.0
    dropout: float = 0.1
    initial_train_size: int = 400
    test_window: int = 60
    walk_step: int | None = None
    seed: int = 42
    device: str = "cpu"
    save_results: bool = True
    output_dir: str = "src/LSTM/models"
    save_model: bool = True

    def resolved_walk_step(self) -> int:
        return self.test_window if self.walk_step is None else self.walk_step
    
    def get_csv_path(self) -> str:
        """Retorna el CSV path, generándolo desde JSONL si es necesario."""
        if self.csv_path:
            return self.csv_path
        
        if not self.jsonl_path:
            raise ValueError("Debe especificar --csv-path o --jsonl-path")
        
        # Generar CSV desde JSONL usando training_data
        from src.LSTM.training_data import combine_sentiment_and_financial
        
        # Crear CSV temporal o en directorio del proyecto
        import tempfile
        temp_csv = Path(tempfile.gettempdir()) / f"lstm_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        print(f"Generando CSV desde JSONL: {self.jsonl_path}")
        combine_sentiment_and_financial(
            sentiment_jsonl_path=self.jsonl_path,
            output_csv_path=str(temp_csv),
            merval_ticker=self.merval_ticker
        )
        print(f"CSV generado: {temp_csv}")
        return str(temp_csv)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BooleanLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


def build_sequences(features: np.ndarray, labels: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    if lookback <= 0:
        raise ValueError("lookback debe ser mayor a 0.")
    if len(features) <= lookback:
        raise ValueError("No hay suficientes observaciones para crear ventanas.")

    sequences: List[np.ndarray] = []
    targets: List[float] = []
    for idx in range(lookback, len(features)):
        sequences.append(features[idx - lookback : idx])
        targets.append(labels[idx])

    X_seq = np.stack(sequences).astype(np.float64)
    y_seq = np.array(targets, dtype=np.float64)
    return X_seq, y_seq


def iter_minibatches(
    tensor_x: torch.Tensor, tensor_y: torch.Tensor, batch_size: int, shuffle: bool = True
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    for batch in loader:
        yield batch


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    target_col = "booleano_merval"
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en {csv_path}")
    feature_cols = [col for col in df.columns if col != target_col]
    features = df[feature_cols].to_numpy(dtype=np.float64)
    labels = df[target_col].astype(int).to_numpy(dtype=np.float64)
    return features, labels, feature_cols


def scale_windows(
    train_windows: np.ndarray, test_windows: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    num_features = train_windows.shape[-1]
    scaler = StandardScaler()
    train_flat = train_windows.reshape(-1, num_features)
    scaler.fit(train_flat)
    train_scaled = scaler.transform(train_flat).reshape(train_windows.shape)
    test_flat = test_windows.reshape(-1, num_features)
    test_scaled = scaler.transform(test_flat).reshape(test_windows.shape)
    return train_scaled, test_scaled, scaler


def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainingConfig,
) -> None:
    

    # Balanceo de clases 
    device = torch.device(config.device)
    model.to(device)

    # Calcular pos_weight para balancear clases
    n_negative = float((y_train == 0).sum())
    n_positive = float((y_train == 1).sum())

    if n_positive > 0 and n_negative > 0:
        # pos_weight = n_negative / n_positive
        # Si hay más negativos, pos_weight > 1 (penaliza más errores en positivos)
        # Si hay más positivos, pos_weight < 1 (penaliza más errores en negativos)
        pos_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32).to(device)
        print(f"Balanceo de clases: Negativos={n_negative:.0f}, Positivos={n_positive:.0f}, pos_weight={pos_weight.item():.3f}")
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    tensor_x = torch.from_numpy(X_train.astype(np.float32))
    tensor_y = torch.from_numpy(y_train.astype(np.float32))

    for epoch in range(1, config.epochs + 1):
        epoch_losses: List[float] = []
        model.train()
        for batch_x, batch_y in iter_minibatches(tensor_x, tensor_y, config.batch_size):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if config.clip_value is not None and config.clip_value > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        print(f"Epoch {epoch}/{config.epochs} - Loss: {avg_loss:.4f}")


def evaluate_model(
    model: nn.Module, 
    X: np.ndarray, 
    y_true: np.ndarray, 
    device: torch.device,
    return_predictions: bool = False
) -> Dict:
    """Evalua el modelo y retorna metricas."""
    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(tensor_x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    y_flat = y_true.astype(int)
    
    result = {
        "accuracy": accuracy_score(y_flat, preds),
        "precision": precision_score(y_flat, preds, zero_division=0),
        "recall": recall_score(y_flat, preds, zero_division=0),
        "f1": f1_score(y_flat, preds, zero_division=0),
    }
    
    if return_predictions:
        result.update({
            "probabilities": probs,
            "predictions": preds,
            "y_true": y_flat,
            "logits": logits.cpu().numpy().flatten()
        })
    
    return result


def walk_forward_training(
    features: np.ndarray, 
    labels: np.ndarray, 
    config: TrainingConfig,
    return_predictions: bool = False
) -> Tuple[List[Dict], Optional[nn.Module], Optional[List[Dict]]]:
    """Entrena modelo con walk-forward validation."""
    sequences, targets = build_sequences(features, labels, config.lookback)
    total_sequences = len(sequences)

    if total_sequences <= config.test_window:
        raise ValueError("No hay suficientes datos para construir el test_window solicitado.")

    initial_train = min(
        max(config.initial_train_size, config.lookback),
        total_sequences - config.test_window,
    )
    if initial_train <= 0 or initial_train >= total_sequences:
        raise ValueError("Tamaño de entrenamiento inicial inválido para la cantidad de datos disponible.")

    walk_step = max(1, config.resolved_walk_step())
    fold_metrics: List[Dict] = []
    fold_predictions: List[Dict] = []
    fold_idx = 1
    start = initial_train
    device = torch.device(config.device)
    last_model: Optional[nn.Module] = None

    while start + config.test_window <= total_sequences:
        train_slice = slice(0, start)
        test_slice = slice(start, start + config.test_window)
        X_train_raw = sequences[train_slice]
        y_train = targets[train_slice]
        X_test_raw = sequences[test_slice]
        y_test = targets[test_slice]

        X_train, X_test, _ = scale_windows(X_train_raw, X_test_raw)
        model = BooleanLSTM(input_dim=X_train.shape[-1], hidden_dim=config.hidden_size, dropout=config.dropout)

        print(
            f"\n=== Fold {fold_idx} | Train seq: {len(X_train)} | Test seq: {len(X_test)} | "
            f"Inicio test idx: {start} ==="
        )
        train_one_fold(model, X_train, y_train, config)
        metrics = evaluate_model(model.to(device), X_test, y_test, device, return_predictions=return_predictions)
        print(
            f"Fold {fold_idx} - Acc: {metrics['accuracy']:.3f} "
            f"Prec: {metrics['precision']:.3f} Rec: {metrics['recall']:.3f} "
            f"F1: {metrics['f1']:.3f}"
        )
        fold_metrics.append(metrics)
        
        # Guardar predicciones si están disponibles
        if return_predictions and "y_true" in metrics:
            fold_predictions.append({
                "y_true": metrics["y_true"],
                "predictions": metrics["predictions"],
                "probabilities": metrics["probabilities"],
            })
        
        last_model = model
        fold_idx += 1
        start += walk_step

    if not fold_metrics:
        raise RuntimeError("No se pudo crear ningún fold de walk-forward. Ajusta test_window o initial_train_size.")
    
    predictions_data = fold_predictions if fold_predictions else None
    return fold_metrics, last_model, predictions_data


def summarize_metrics(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key in metrics[0].keys():
        if key not in ["probabilities", "predictions", "y_true", "logits"]:
            values = [fold[key] for fold in metrics if key in fold]
            if values:
                summary[key] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
    return summary


def generate_performance_plots(
    fold_metrics: List[Dict],
    predictions_data: Optional[List[Dict]],
    model_dir: Path,
    config: TrainingConfig
) -> None:
    """Genera graficos basicos de performance."""
    if not predictions_data:
        return
    
    all_y_true = []
    all_y_pred = []
    
    for fold_preds in predictions_data:
        if "y_true" in fold_preds and "predictions" in fold_preds:
            all_y_true.extend(fold_preds["y_true"].tolist() if isinstance(fold_preds["y_true"], np.ndarray) else fold_preds["y_true"])
            all_y_pred.extend(fold_preds["predictions"].tolist() if isinstance(fold_preds["predictions"], np.ndarray) else fold_preds["predictions"])
    
    if all_y_true and all_y_pred:
        cm = confusion_matrix(all_y_true, all_y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Baja', 'Sube'], yticklabels=['Baja', 'Sube'])
        ax.set_title('Matriz de Confusion')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
        plt.tight_layout()
        plt.savefig(model_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()


def save_training_results(
    config: TrainingConfig,
    fold_metrics: List[Dict],
    feature_cols: List[str],
    total_samples: int,
    predictions_data: Optional[List[Dict]] = None,
    model: Optional[nn.Module] = None,
    output_dir: Optional[Path] = None,
    jsonl_source: Optional[str] = None
) -> Path:
    """Guarda resultados del entrenamiento."""
    if output_dir is None:
        output_dir = Path(config.output_dir)
    else:
        output_dir = Path(output_dir)
    
    # Crear directorio con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mean_acc = np.mean([m.get("accuracy", 0) for m in fold_metrics if "accuracy" in m])
    model_dir = output_dir / f"{timestamp}_acc{mean_acc:.3f}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Guardando resultados en: {model_dir}")
    
    # 1. Guardar configuración
    config_dict = {
        "timestamp": timestamp,
        "csv_path": config.csv_path,
        "jsonl_source": jsonl_source,  # Ruta al JSONL procesado usado (si aplica)
        "hyperparameters": {
            "lookback": config.lookback,
            "hidden_size": config.hidden_size,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "clip_value": config.clip_value,
            "dropout": config.dropout,
            "seed": config.seed,
            "device": config.device,
        },
        "walk_forward": {
            "initial_train_size": config.initial_train_size,
            "test_window": config.test_window,
            "walk_step": config.resolved_walk_step(),
        },
        "dataset_info": {
            "total_samples": total_samples,
            "num_features": len(feature_cols),
            "features": feature_cols,
        },
    }
    
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # 2. Guardar métricas por fold
    summary = summarize_metrics(fold_metrics)
    
    metrics_dict = {
        "folds": [
            {
                "fold": idx + 1,
                "metrics": {
                    k: float(v) for k, v in fold.items()
                    if k not in ["probabilities", "predictions", "y_true", "logits"]
                }
            }
            for idx, fold in enumerate(fold_metrics)
        ],
        "summary": summary,
    }
    
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    
    # 3. Guardar predicciones y probabilidades si están disponibles
    if predictions_data:
        all_predictions = []
        for fold_idx, fold_preds in enumerate(predictions_data):
            if "y_true" in fold_preds and "predictions" in fold_preds:
                for i in range(len(fold_preds["y_true"])):
                    all_predictions.append({
                        "fold": fold_idx + 1,
                        "y_true": int(fold_preds["y_true"][i]),
                        "y_pred": int(fold_preds["predictions"][i]),
                        "probabilidad": float(fold_preds["probabilities"][i]),
                        "correcto": bool(fold_preds["y_true"][i] == fold_preds["predictions"][i]),
                    })
        
        if all_predictions:
            df_predictions = pd.DataFrame(all_predictions)
            df_predictions.to_csv(model_dir / "predictions.csv", index=False)
    
    # 4. Guardar resumen en formato JSONL para historial
    history_entry = {
        "timestamp": timestamp,
        "mean_accuracy": float(mean_acc),
        "mean_f1": float(summary.get("f1", 0)),
        "mean_precision": float(summary.get("precision", 0)),
        "mean_recall": float(summary.get("recall", 0)),
        "num_folds": len(fold_metrics),
        "model_dir": str(model_dir),
        "config_hash": f"{config.lookback}_{config.hidden_size}_{config.learning_rate}",
    }
    
    history_file = output_dir / "training_history.jsonl"
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")
    
    # 5. Guardar modelo si se proporciona
    if model is not None and config.save_model:
        model_path = model_dir / "model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config_dict,
            "summary": summary,
        }, model_path)
        print(f"Modelo guardado: {model_path}")
    
    print(f"Configuracion guardada: {model_dir / 'config.json'}")
    print(f"Metricas guardadas: {model_dir / 'metrics.json'}")
    if predictions_data:
        print(f"Predicciones guardadas: {model_dir / 'predictions.csv'}")
    print(f"Historial actualizado: {history_file}")
    
    # 6. Generar y guardar gráficos de performance
    try:
        generate_performance_plots(fold_metrics, predictions_data, model_dir, config)
    except Exception as e:
        print(f"Error al generar graficos: {e}")
    
    return model_dir


def prepare_dataset_from_jsonl(
    jsonl_path: str,
    output_csv_path: Optional[str] = None,
    merval_ticker: str = "^MERV"
) -> str:
    """
    Prepara dataset CSV desde JSONL procesado usando training_data.
    
    Args:
        jsonl_path: Ruta al JSONL procesado (con sentimiento)
        output_csv_path: Ruta donde guardar CSV (si None, se genera automáticamente)
        merval_ticker: Ticker de MERVAL
    
    Returns:
        Ruta al CSV generado
    """
    # Importar aquí para evitar dependencias circulares
    from src.LSTM.training_data import combine_sentiment_and_financial
    
    if output_csv_path is None:
        # Generar nombre automático basado en el JSONL
        jsonl_stem = Path(jsonl_path).stem
        output_csv_path = f"src/LSTM/dataset_{jsonl_stem}.csv"
    
    print(f"Generando dataset CSV desde JSONL procesado...")
    print(f"   JSONL: {jsonl_path}")
    print(f"   CSV: {output_csv_path}")
    
    combine_sentiment_and_financial(
        sentiment_jsonl_path=jsonl_path,
        output_csv_path=output_csv_path,
        merval_ticker=merval_ticker
    )
    
    return output_csv_path


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Entrena una LSTM en PyTorch con walk-forward.",
        epilog="""
Ejemplos:
  # Entrenar desde JSONL procesado (recomendado - busca automáticamente el último)
  python train_boolean_lstm.py --jsonl-auto
  
  # Entrenar desde JSONL específico
  python train_boolean_lstm.py --jsonl-path data/procesada/unified_data_20251112_with_sentiment.jsonl
  
  # Entrenar desde CSV existente
  python train_boolean_lstm.py --csv-path src/LSTM/dataset_con_sentiment.csv
        """
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Ruta al CSV con datos preparados")
    parser.add_argument("--jsonl-path", type=str, default=None, help="Ruta al JSONL procesado (con sentimiento). Genera CSV automáticamente.")
    parser.add_argument("--jsonl-auto", action="store_true", help="Buscar automáticamente el último JSONL en data/procesada/")
    parser.add_argument("--merval-ticker", type=str, default="^MERV", help="Ticker de MERVAL (default: ^MERV)")
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-value", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--initial-train-size", type=int, default=20)
    parser.add_argument("--test-window", type=int, default=5)
    parser.add_argument("--walk-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-results", action="store_true", default=True, help="Guardar resultados automáticamente (default: True)")
    parser.add_argument("--no-save-results", dest="save_results", action="store_false", help="No guardar resultados")
    parser.add_argument("--output-dir", type=str, default="src/LSTM/models", help="Directorio donde guardar resultados (default: src/LSTM/models)")
    parser.add_argument("--save-model", action="store_true", default=True, help="Guardar checkpoint del modelo (default: True)")
    parser.add_argument("--no-save-model", dest="save_model", action="store_false", help="No guardar checkpoint del modelo")
    args = parser.parse_args()
    
    # Determinar jsonl_path
    jsonl_path = None
    if args.jsonl_auto:
        from src.LSTM.training_data import find_latest_processed_jsonl
        jsonl_path = find_latest_processed_jsonl()
        if jsonl_path is None:
            raise ValueError(
                "No se encontró ningún archivo JSONL en data/procesada/\n"
                "Ejecuta primero: python src/procesamiento/run_sentiment.py <archivo_preprocesado>"
            )
        print(f"JSONL encontrado: {jsonl_path}")
    elif args.jsonl_path:
        jsonl_path = args.jsonl_path
    
    # Si no se especifica ni CSV ni JSONL, intentar buscar JSONL automáticamente
    if not args.csv_path and not jsonl_path:
        from src.LSTM.training_data import find_latest_processed_jsonl
        jsonl_path = find_latest_processed_jsonl()
        if jsonl_path:
            print(f"Usando ultimo JSONL encontrado: {jsonl_path}")
        else:
            args.csv_path = "src/LSTM/dataset_con_sentiment.csv"
            print(f"Usando CSV por defecto: {args.csv_path}")
    
    return TrainingConfig(
        csv_path=args.csv_path,
        jsonl_path=jsonl_path,
        merval_ticker=args.merval_ticker,
        lookback=args.lookback,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_value=args.clip_value,
        dropout=args.dropout,
        initial_train_size=args.initial_train_size,
        test_window=args.test_window,
        walk_step=args.walk_step,
        seed=args.seed,
        device=args.device,
        save_results=args.save_results,
        output_dir=args.output_dir,
        save_model=args.save_model,
    )


def main() -> None:
    config = parse_args()
    set_global_seed(config.seed)
    device = torch.device(config.device)
    print(f"Usando dispositivo: {device}")
    
    # Obtener CSV path (generándolo desde JSONL si es necesario)
    csv_path = config.get_csv_path()
    print(f"Usando dataset: {csv_path}")
    
    features, labels, feature_cols = load_dataset(csv_path)
    print(f"Columnas de features: {feature_cols}")
    
    # Entrenar con predicciones si vamos a guardar resultados
    return_predictions = config.save_results
    metrics, last_model, predictions_data = walk_forward_training(
        features, labels, config, return_predictions=return_predictions
    )
    
    summary = summarize_metrics(metrics)
    print("\n==== Resumen Walk-Forward ====")
    for key, value in summary.items():
        if not key.endswith("_std"):
            print(f"{key.capitalize()}: {value:.3f}")
    
    # Guardar resultados si está habilitado
    if config.save_results:
        model_dir = save_training_results(
            config=config,
            fold_metrics=metrics,
            feature_cols=feature_cols,
            total_samples=len(features),
            predictions_data=predictions_data,
            model=last_model,
            output_dir=Path(config.output_dir),
            jsonl_source=config.jsonl_path,
        )
        print(f"Resultados guardados en: {model_dir}")


if __name__ == "__main__":
    main()
