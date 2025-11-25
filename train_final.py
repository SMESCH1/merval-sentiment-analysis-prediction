import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from model_utils import MaskedSeqDataset, LSTMBinary

# ==========================================
# CONFIGURACION Y PARAMETROS
# ==========================================
DATA_PATH       = 'data/data_train.csv'
MODEL_SAVE_PATH = 'data/final_model_state.pth'
LOSS_PLOT_PATH  = 'data/loss_curve_final.png'


# Parametros de Datos
SEQ_LEN         = 90        # Longitud de ventana (dias de historia)

# Hiperparametros del Modelo
BATCH_SIZE      = 16
HIDDEN_SIZE     = 28
NUM_LAYERS      = 1
DROPOUT         = 0.46
LR              = 0.00057
N_EPOCHS        = 100
BIDIRECTIONAL   = False
DEFAULT_THRESHOLD = 0.3


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    print(f"Usando dispositivo: {DEVICE}")
    
    # 1. Carga de Datos
    print(f"Datos desde {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep=';')
    
    feature_cols = [c for c in df.columns if c != 'prediccion']

    # Preparar X e y con todos los datos
    y_full = df['prediccion'].astype('float32').to_numpy()
    X_full = df[feature_cols].to_numpy(dtype='float32')
    
    print(f"Entrenando con {len(X_full)} registros.")
    
    # 2. Normalizacion de los atributos considerando todos los datos
    X_mean = X_full.mean(axis=0, keepdims=True)
    X_std = X_full.std(axis=0, keepdims=True)
    X_full = (X_full - X_mean) / (X_std + 1e-8)

    # 3. Dataset y Dataloader
    full_ds = MaskedSeqDataset(X_full, y_full, SEQ_LEN)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Modelo y optimizador
    final_model = LSTMBinary(input_size=len(feature_cols), hidden_size=HIDDEN_SIZE, 
                            num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL).to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(final_model.parameters(), lr=LR)

    epoch_losses = []
    print("Iniciando entrenamiento final...")

    for epoch in range(1, N_EPOCHS + 1):
        final_model.train()
        running_loss = 0.0
        running_count = 0
        for xb, yb in full_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = final_model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
            running_count += 1
            
        avg_loss = running_loss / max(running_count, 1)
        epoch_losses.append(avg_loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{N_EPOCHS} - Loss: {avg_loss:.4f}")

    # 5. Evaluar AUC en el conjunto de entrenamiento
    final_model.eval()
    train_probs = []
    train_labels = []
    with torch.no_grad():
        for xb, yb in full_loader:
            xb = xb.to(DEVICE)
            logits = final_model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            train_probs.append(probs)
            train_labels.append(yb.numpy())
    
    if len(train_probs) > 0:
        train_probs = np.concatenate(train_probs)
        train_labels = np.concatenate(train_labels)
        
        if len(np.unique(train_labels)) > 1:
            # AUC
            train_auc = roc_auc_score(train_labels, train_probs)
            
            # Predicciones binarias con umbral
            train_preds = (train_probs > DEFAULT_THRESHOLD).astype(int)
            
            # Métricas de clasificación
            accuracy = accuracy_score(train_labels, train_preds)
            precision = precision_score(train_labels, train_preds, zero_division=0)
            recall = recall_score(train_labels, train_preds, zero_division=0)
            f1 = f1_score(train_labels, train_preds, zero_division=0)
            
            # Matriz de confusión
            tn, fp, fn, tp = confusion_matrix(train_labels.astype(int), train_preds, labels=[0, 1]).ravel()
            
            # Mostrar resultados
            print("\n" + "="*60)
            print("MÉTRICAS DEL MODELO FINAL (Conjunto de Entrenamiento)")
            print("="*60)
            print(f"AUC:       {train_auc:.4f}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"\nUmbral usado: {DEFAULT_THRESHOLD}")
            print("\n" + "-"*60)
            print("MATRIZ DE CONFUSIÓN")
            print("-"*60)
            print(f"TN: {tn:4d}  |  FP: {fp:4d}")
            print(f"FN: {fn:4d}  |  TP: {tp:4d}")
            print("="*60)
        else:
            print("\nNo se pudo calcular AUC (solo una clase en labels)")
    else:
        print("\nNo se generaron predicciones")

    # 6. Guardar modelo y metadatos
    checkpoint = {
        'model_state_dict': final_model.state_dict(),
        'config': {
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'seq_len': SEQ_LEN,
            'batch_size': BATCH_SIZE,
            'threshold': DEFAULT_THRESHOLD,
            'bidirectional': BIDIRECTIONAL
        },
        'X_mean': X_mean,
        'X_std': X_std,
        'feature_cols': feature_cols
    }
    
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"Modelo y estado guardados en {MODEL_SAVE_PATH}")
    
    # 7. Graficar loss
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.title('Curva de Aprendizaje (Entrenamiento Final)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_PLOT_PATH)
    print(f"Grafico de loss guardado en {LOSS_PLOT_PATH}")

if __name__ == "__main__":
    main()
