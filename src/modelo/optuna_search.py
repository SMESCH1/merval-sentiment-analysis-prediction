import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import optuna
from optuna.trial import Trial
from sklearn.metrics import roc_auc_score
from model_utils import MaskedSeqDataset, rolling_splits, LSTMBinary, train_fold

# ==========================================
# CONFIGURACION Y PARAMETROS
# ==========================================
DATA_PATH       = 'data/data_train.csv'
STUDY_NAME      = 'lstm_stock_optimization'
N_TRIALS        = 200
N_JOBS          = 1  # Paralelizacion (1 = secuencial)

# Parametros de Datos (fijos)
# SEQ_LEN         = 8

# Parametros de Cross-Validation (fijos)
TRAIN_SIZE      = 173
VAL_SIZE        = 30

# Parametros fijos del entrenamiento
# BATCH_SIZE      = 16
DEFAULT_THRESHOLD = 0.5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# FUNCION OBJETIVO PARA OPTUNA
# ==========================================

def objective(trial: Trial):
    """
    Funcion objetivo que Optuna intentara maximizar.
    Retorna el AUC promedio en validacion cruzada.
    """
    
    # 1. Sugerir hiperparametros
    SEQ_LEN = trial.suggest_int('seq_len', 10, 20, step=2)
    BATCH_SIZE = trial.suggest_int('batch_size', 16, 64, step=16)
    
    hidden_size = trial.suggest_int('hidden_size', 16, 64, step=16)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    n_epochs = trial.suggest_int('n_epochs', 50, 200, step=50)
    
    # 1.5 Nuevos parametros fijos
    num_layers = 1
    
    # 2. Cargar datos (esto se hace cada trial, podriamos cachearlo pero es rapido)
    df = pd.read_csv(DATA_PATH, sep=';')
    # Preprocesamiento: Convertir VERDADERO/FALSO a 1/0
    df = df.replace({'VERDADERO': 1, 'FALSO': 0})
    
    # Rellenar NaNs en features con 0
    feature_cols = [c for c in df.columns if c != 'prediccion']
    df[feature_cols] = df[feature_cols].fillna(0)
    
    y_raw = df['prediccion']
    y = y_raw.astype('float32').to_numpy()
    X = df[feature_cols].to_numpy(dtype='float32')
    
    T, n_features = X.shape
    
    # 3. Cross-Validation
    splits = rolling_splits(T, TRAIN_SIZE, VAL_SIZE)
    aucs = []
    
    for i, (tr_s, tr_e, va_s, va_e) in enumerate(splits):
        # Slicing de datos RAW
        X_train_raw, y_train = X[tr_s:tr_e], y[tr_s:tr_e]
        X_val_raw,   y_val   = X[va_s:va_e], y[va_s:va_e]
        
        # Normalizacion (Fit solo en TRAIN)
        X_mean = X_train_raw.mean(axis=0, keepdims=True)
        X_std  = X_train_raw.std(axis=0, keepdims=True)
        
        X_train = (X_train_raw - X_mean) / (X_std + 1e-8)
        X_val   = (X_val_raw - X_mean)   / (X_std + 1e-8)
        
        # Datasets
        try:
            train_ds = MaskedSeqDataset(X_train, y_train, SEQ_LEN)
            val_ds   = MaskedSeqDataset(X_val,   y_val,   SEQ_LEN)
        except ValueError:
            # Si no hay suficientes datos validos, saltar este fold
            continue
        
        # Sampler balanceado
        labels_sampler = torch.tensor(train_ds.y[train_ds.valid_ends], dtype=torch.float32)
        pos_count = (labels_sampler == 1).sum().item()
        neg_count = (labels_sampler == 0).sum().item()
        
        pos_weight = None
        if pos_count > 0 and neg_count > 0:
            pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=DEVICE)
            weights = torch.where(labels_sampler == 1,
                                  1.0 / pos_count,
                                  1.0 / neg_count)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        else:
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Modelo
        model = LSTMBinary(input_size=n_features, hidden_size=hidden_size, 
                          num_layers=num_layers, dropout=dropout)
        
        # Entrenar
        model.to(DEVICE)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        
        for _ in range(n_epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
        
        # Calcular AUC en validacion
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs)
                val_labels.append(yb.numpy())
        
        if len(val_probs) > 0:
            val_probs = np.concatenate(val_probs)
            val_labels = np.concatenate(val_labels)
            
            # Calcular AUC solo si hay ambas clases
            if len(np.unique(val_labels)) > 1:
                auc = roc_auc_score(val_labels, val_probs)
                aucs.append(auc)
        
        # Reportar AUC intermedio para pruning
        if len(aucs) > 0:
            trial.report(np.mean(aucs), i)
        
        # Optuna puede detener trials que no son prometedores
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # 4. Retornar metrica objetivo (AUC promedio)
    if len(aucs) == 0:
        return 0.0
    
    return np.mean(aucs)


# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    import sys
    
    # Permitir pasar N_TRIALS como argumento de línea de comandos
    n_trials = N_TRIALS
    if len(sys.argv) > 1:
        try:
            n_trials = int(sys.argv[1])
        except ValueError:
            print(f"Advertencia: '{sys.argv[1]}' no es un número válido. Usando N_TRIALS={N_TRIALS}")
            n_trials = N_TRIALS
    
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Iniciando busqueda de hiperparametros con Optuna...")
    print(f"N_TRIALS: {n_trials}")
    
    # Crear o cargar estudio
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='maximize',  # Queremos maximizar el AUC
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage='sqlite:///data/optuna_study.db',  # Base de datos compartida para paralelizacion
        load_if_exists=True  # Reutilizar estudio existente si ya existe
    )
    
    # Ejecutar optimizacion
    study.optimize(objective, n_trials=n_trials, n_jobs=N_JOBS, show_progress_bar=True)
    
    # Resultados
    print("\n" + "="*60)
    print("OPTIMIZACION COMPLETADA")
    print("="*60)
    print(f"Mejor AUC: {study.best_value:.4f}")
    print("\nMejores hiperparametros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Guardar resultados
    df_trials = study.trials_dataframe()
    df_trials.to_csv('data/optuna_trials.csv', index=False)
    print(f"\nHistorial de trials guardado en: data/optuna_trials.csv")
    
    # Guardar mejores parametros
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df.to_csv('data/best_params.csv', index=False)
    print(f"Mejores parametros guardados en: data/best_params.csv")
    
    # Grafico de importancia de parametros
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig('data/param_importance.png', dpi=150)
        plt.close()
        print(f"Grafico de importancia guardado en: data/param_importance.png")
    except Exception as e:
        print(f"No se pudo generar grafico de importancia: {e}")
    
    # Historial de optimizacion
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig('data/optimization_history.png', dpi=150)
        plt.close()
        print(f"Historial de optimizacion guardado en: data/optimization_history.png")
    except Exception as e:
        print(f"No se pudo generar historial: {e}")


if __name__ == "__main__":
    main()
