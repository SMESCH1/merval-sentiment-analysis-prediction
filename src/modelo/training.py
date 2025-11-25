import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from .model_utils import MaskedSeqDataset, rolling_splits, LSTMBinary, train_fold

# ==========================================
# CONFIGURACION Y PARAMETROS

DATA_PATH       = 'src/modelo/data/data_train.csv'
VAL_PREDS_PATH  = 'src/modelo/data/val_predictions.csv'

# Parametros de Cross-Validation (Rolling)
TRAIN_SIZE      = 650
VAL_SIZE        = 100

# Parametros de Datos
SEQ_LEN         = 90        # Longitud de ventana (dias de historia)

# Hiperparametros del Modelo
BATCH_SIZE      = 16
HIDDEN_SIZE     = 28
NUM_LAYERS      = 1
DROPOUT         = 0.46
LR              = 0.00057
N_EPOCHS        = 40
BIDIRECTIONAL   = False
DEFAULT_THRESHOLD = 0.5

# Early Stopping
PATIENCE        = 70     # Epochs sin mejora antes de detener (aumentado)
MIN_DELTA       = 0.001  # Mínima mejora considerada progreso (más tolerante)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# SCRIPT PRINCIPAL


def main():
    print(f"Usando dispositivo: {DEVICE}")
    
    # 1. Carga de Datos
    print(f"Datos desde {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep=';')
    
    # Rellenar NaNs en features con 0. 
    # Por como está armado el dataset, no debería haber, pero igualmente, esto
    # implicaría algo conservador, que es que ese día no hubo variaciones en ese atributo.
    feature_cols = [c for c in df.columns if c != 'prediccion']
    
    # Preparar X e y
    y_raw = df['prediccion']
    y = y_raw.astype('float32').to_numpy()
    X = df[feature_cols].to_numpy(dtype='float32')
    
    print(f"Dimensiones de entrenamiento (CV): {X.shape}")

    # 2. Definir folds de Cross-Validation 
    T, n_features = X.shape
    splits = rolling_splits(T, TRAIN_SIZE, VAL_SIZE)
    
    # Listas para guardar AUC de cada fold
    aucs = []
    best_epochs = []  # Track de mejores epochs para análisis
    
    # Listas para guardar predicciones de validacion
    all_val_probs = [] #Output de la red para cada elto de validación
    all_val_labels = [] #Etiquetas reales de validación

    #Bucle de Cross-Validation. Se entrena un fold y se evalúa.   
    for i, (tr_s, tr_e, va_s, va_e) in enumerate(splits, start=1):
        # Slicing de datos RAW
        X_train_raw, y_train = X[tr_s:tr_e], y[tr_s:tr_e]
        X_val_raw,   y_val   = X[va_s:va_e], y[va_s:va_e]
        
        # Normalizacion de los atributos sólo considerando la información del fold.
        X_mean = X_train_raw.mean(axis=0, keepdims=True)
        X_std  = X_train_raw.std(axis=0, keepdims=True)
        
        X_train = (X_train_raw - X_mean) / (X_std + 1e-8)
        X_val   = (X_val_raw - X_mean)   / (X_std + 1e-8)

        # Datasets
        train_ds = MaskedSeqDataset(X_train, y_train, SEQ_LEN)
        val_ds   = MaskedSeqDataset(X_val,   y_val,   SEQ_LEN)
        
        # Dataloaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Modelo
        model = LSTMBinary(input_size=n_features, hidden_size=HIDDEN_SIZE, 
                           num_layers=NUM_LAYERS, dropout=DROPOUT, 
                           bidirectional=BIDIRECTIONAL)
        
        # Entrenar fold con early stopping
        best_auc, best_epoch = train_fold(model, train_loader, val_loader, 
                                          n_epochs=N_EPOCHS, lr=LR, device=DEVICE, 
                                          threshold=DEFAULT_THRESHOLD,
                                          patience=PATIENCE, min_delta=MIN_DELTA)
        
        aucs.append(best_auc)
        best_epochs.append(best_epoch)

        print(f"Fold {i} - Best val AUC: {best_auc:.4f} (epoch {best_epoch}/{N_EPOCHS})")

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
    print(f"Mean AUC: {np.mean(aucs):.4f}")
    
    # Estadísticas de Early Stopping
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE EARLY STOPPING")
    print("="*60)
    print(f"Epoch promedio óptimo: {np.mean(best_epochs):.1f}")
    print(f"Rango de epochs: {min(best_epochs)} - {max(best_epochs)}")
    print(f"Recomendación para train_final: N_EPOCHS = {int(np.mean(best_epochs))}")
    print("="*60)

    # Guardar CSV acumulado de validacion
    if len(all_val_probs) > 0:
        probs_all = np.concatenate(all_val_probs)
        labels_all = np.concatenate(all_val_labels)
        val_preds_df = pd.DataFrame({"prob": probs_all, "label": labels_all})
        val_preds_df.to_csv(VAL_PREDS_PATH, index=False)
        print(f"Predicciones de validacion guardadas en: {VAL_PREDS_PATH}")
        
        # AUC global
        auc_global = roc_auc_score(labels_all, probs_all)
        print(f"AUC Global (acumulado): {auc_global:.4f}")
        
        # Curva ROC
        fpr, tpr, thresholds = roc_curve(labels_all, probs_all)
        
        # Graficar curva ROC
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_global:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
        plt.title('Curva ROC - Validación Cruzada', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        from pathlib import Path
        roc_path = Path('src/modelo/data/roc_curve.png')
        roc_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(roc_path), dpi=150)
        print(f"Curva ROC guardada en: {roc_path}")
        plt.close()
        
        # Analisis de umbrales
        print("\n" + "="*60)
        print("ANÁLISIS DE DIFERENTES UMBRALES")
        print("="*60)
        test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        for thresh in test_thresholds:
            y_pred = (probs_all > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels_all.astype(int), y_pred, labels=[0, 1]).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nUmbral: {thresh:.2f}")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  TN: {tn:3d} | FP: {fp:3d} | FN: {fn:3d} | TP: {tp:3d}")
        
        # Matriz de confusion con umbral DEFAULT
        print("\n" + "="*60)
        print(f"MATRIZ DE CONFUSIÓN CON UMBRAL {DEFAULT_THRESHOLD}")
        print("="*60)
        y_pred_val = (probs_all > DEFAULT_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels_all.astype(int), y_pred_val, labels=[0, 1]).ravel()
        print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")
    else:
        print("No se generaron predicciones de validacion.")

if __name__ == "__main__":
    main()
