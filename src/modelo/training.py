import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model_utils import MaskedSeqDataset, rolling_splits, LSTMBinary, train_fold

# ==========================================
# CONFIGURACION Y PARAMETROS
# ==========================================
DATA_PATH       = 'data/data_train.csv'
VAL_PREDS_PATH  = 'data/val_predictions.csv'

# Parametros de Cross-Validation (Rolling)
TRAIN_SIZE      = 134       # Puntos base en cada ventana de entrenamiento
VAL_SIZE        = 30        # Puntos base en cada ventana de validacion

# Parametros de Datos
SEQ_LEN         = 12        # Longitud de ventana (dias de historia)

# Hiperparametros del Modelo
BATCH_SIZE      = 16
HIDDEN_SIZE     = 64
NUM_LAYERS      = 1
DROPOUT         = 0.35
LR              = 0.001
N_EPOCHS        = 200
DEFAULT_THRESHOLD = 0.5


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    print(f"Usando dispositivo: {DEVICE}")
    
    # 1. Carga de Datos
    print(f"Cargando datos desde {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep=';')
    
    # Preprocesamiento: Convertir VERDADERO/FALSO a 1/0
    df = df.replace({'VERDADERO': 1, 'FALSO': 0})
    
    # Rellenar NaNs en features con 0
    feature_cols = [c for c in df.columns if c != 'prediccion']
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Preparar X e y
    y_raw = df['prediccion']
    y = y_raw.astype('float32').to_numpy()
    X = df[feature_cols].to_numpy(dtype='float32')
    
    print(f"Dimensiones de entrenamiento (CV): {X.shape}")

    # 2. Definir Splits de Cross-Validation (sin normalizar aun)
    T, n_features = X.shape
    splits = rolling_splits(T, TRAIN_SIZE, VAL_SIZE)
    print(f"Total folds generados: {len(splits)}")
    
    accs = []
    precisions = []
    all_val_probs = []
    all_val_labels = []

    # 3. Bucle de Entrenamiento (Cross-Validation)
    print("Iniciando Cross-Validation...")
    
    for i, (tr_s, tr_e, va_s, va_e) in enumerate(splits, start=1):
        # Slicing de datos RAW
        X_train_raw, y_train = X[tr_s:tr_e], y[tr_s:tr_e]
        X_val_raw,   y_val   = X[va_s:va_e], y[va_s:va_e]
        
        # Normalizacion (Fit solo en TRAIN)
        X_mean = X_train_raw.mean(axis=0, keepdims=True)
        X_std  = X_train_raw.std(axis=0, keepdims=True)
        
        X_train = (X_train_raw - X_mean) / (X_std + 1e-8)
        X_val   = (X_val_raw - X_mean)   / (X_std + 1e-8)

        # Datasets
        train_ds = MaskedSeqDataset(X_train, y_train, SEQ_LEN)
        val_ds   = MaskedSeqDataset(X_val,   y_val,   SEQ_LEN)

        # Sampler Balanceado
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
        model = LSTMBinary(input_size=n_features, hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
        
        # Entrenar Fold
        acc, prec = train_fold(model, train_loader, val_loader, 
                               n_epochs=N_EPOCHS, lr=LR, device=DEVICE, 
                               pos_weight=pos_weight, threshold=DEFAULT_THRESHOLD)
        
        accs.append(acc)
        precisions.append(prec)
        print(f"Fold {i} - val acc: {acc:.3f} | val precision: {prec:.3f}")

        # Guardar predicciones de validacion
        val_probs_batches, val_labels_batches = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs_batches.append(probs)
                val_labels_batches.append(yb.numpy())
        if len(val_probs_batches) > 0:
            all_val_probs.append(np.concatenate(val_probs_batches))
            all_val_labels.append(np.concatenate(val_labels_batches))

    # 5. Resultados Finales
    print("\nResultados Promedio Cross-Validation:")
    print(f"Mean Accuracy:  {np.mean(accs):.4f}")
    print(f"Mean Precision: {np.mean(precisions):.4f}")

    # Guardar CSV acumulado de validacion
    if len(all_val_probs) > 0:
        probs_all = np.concatenate(all_val_probs)
        labels_all = np.concatenate(all_val_labels)
        val_preds_df = pd.DataFrame({"prob": probs_all, "label": labels_all})
        val_preds_df.to_csv(VAL_PREDS_PATH, index=False)
        print(f"Predicciones de validacion guardadas en: {VAL_PREDS_PATH}")
        
        # Matriz de confusion global
        y_pred_val = (probs_all > DEFAULT_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels_all.astype(int), y_pred_val, labels=[0, 1]).ravel()
        print("Matriz de Confusion Acumulada (Validacion):")
        print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")
    else:
        print("No se generaron predicciones de validacion.")

if __name__ == "__main__":
    main()
