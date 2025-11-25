import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import optuna
from optuna.trial import Trial
from sklearn.metrics import roc_auc_score
from .model_utils import MaskedSeqDataset, rolling_splits, LSTMBinary, train_fold

# ==========================================
# CONFIGURACION Y PARAMETROS

DATA_PATH       = 'src/modelo/data/data_train.csv'
STUDY_NAME      = 'lstm_stock_optimization'
N_TRIALS        = 200
N_JOBS          = 1  # Paralelizacion (1 = secuencial)

# Parametros de Datos (fijos)
# SEQ_LEN         = 8

# Parametros de Cross-Validation (fijos)
TRAIN_SIZE      = 650
VAL_SIZE        = 100

# Parametros fijos del entrenamiento
# BATCH_SIZE      = 16
DEFAULT_THRESHOLD = 0.5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# FUNCION OBJETIVO PARA OPTUNA

def objective(trial: Trial):
    """
    Funcion objetivo: retorna AUC promedio en CV.
    """

    # Parametros que quedaron fijos
    num_layers = 1
    
    # 1. Hiperparametros a optimizar
    SEQ_LEN = trial.suggest_int('seq_len', 70, 100, step=7)
    BATCH_SIZE = trial.suggest_int('batch_size', 16, 64, step=16)
    hidden_size = trial.suggest_int('hidden_size', 4, 42, step=4)
    dropout = trial.suggest_float('dropout', 0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    n_epochs = trial.suggest_int('n_epochs', 50, 100, step=25)
    bidirectional = trial.suggest_categorical('bidirectional', [False, True])
    
    # 2. Cargar datos
    df = pd.read_csv(DATA_PATH, sep=';')
    feature_cols = [c for c in df.columns if c != 'prediccion']
    
    y = df['prediccion'].astype('float32').to_numpy()
    X = df[feature_cols].to_numpy(dtype='float32')
    T, n_features = X.shape
    
    # 3. Cross-validation
    # Genero los splits
    splits = rolling_splits(T, TRAIN_SIZE, VAL_SIZE)
    aucs = []
    
    for i, (tr_s, tr_e, va_s, va_e) in enumerate(splits):
        X_train_raw, y_train = X[tr_s:tr_e], y[tr_s:tr_e]
        X_val_raw, y_val = X[va_s:va_e], y[va_s:va_e]
        
        # Normalizar solo con datos de train para evitar leakage.
        X_mean = X_train_raw.mean(axis=0, keepdims=True)
        X_std = X_train_raw.std(axis=0, keepdims=True)
        
        X_train = (X_train_raw - X_mean) / (X_std + 1e-8)
        X_val = (X_val_raw - X_mean) / (X_std + 1e-8)
        
        try:
            train_ds = MaskedSeqDataset(X_train, y_train, SEQ_LEN)
            val_ds = MaskedSeqDataset(X_val, y_val, SEQ_LEN)
        except ValueError:
            continue
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Modelo y entrenamiento
        model = LSTMBinary(input_size=n_features, hidden_size=hidden_size, 
                          num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        model.to(DEVICE)
        crit = nn.BCEWithLogitsLoss()
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
        
        # Evaluar en validacion
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
            
            if len(np.unique(val_labels)) > 1:
                auc = roc_auc_score(val_labels, val_probs)
                aucs.append(auc)
        
        # Reportar progreso
        if len(aucs) > 0:
            trial.report(np.mean(aucs), i)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # 4. Retornar metrica (AUC promedio)
    if len(aucs) == 0:
        return 0.0
    return np.mean(aucs)


# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Busqueda de hiperparametros con Optuna")
    print(f"Trials: {N_TRIALS}")
    
    # Crear o cargar estudio
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='maximize',  # Queremos maximizar el AUC
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage='sqlite:///src/modelo/data/optuna_study.db',  # Base de datos compartida para paralelizacion
        load_if_exists=True  # Reutilizar estudio existente si ya existe
    )
    
    # Ejecutar optimizacion
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, show_progress_bar=True)
    
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
    df_trials.to_csv('src/modelo/data/optuna_trials.csv', index=False)
    print(f"\nHistorial de trials guardado en: src/modelo/data/optuna_trials.csv")
    
    # Guardar mejores parametros
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df.to_csv('src/modelo/data/best_params.csv', index=False)
    print(f"Mejores parametros guardados en: src/modelo/data/best_params.csv")
    
    # Grafico de importancia de parametros
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig('src/modelo/data/param_importance.png', dpi=150)
        plt.close()
        print(f"Grafico de importancia guardado en: src/modelo/data/param_importance.png")
    except Exception as e:
        print(f"No se pudo generar grafico de importancia: {e}")
    
    # Historial de optimizacion
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig('src/modelo/data/optimization_history.png', dpi=150)
        plt.close()
        print(f"Historial de optimizacion guardado en: src/modelo/data/optimization_history.png")
    except Exception as e:
        print(f"No se pudo generar historial: {e}")


if __name__ == "__main__":
    main()
