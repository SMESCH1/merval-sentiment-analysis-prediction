import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import MaskedSeqDataset, LSTMBinary

# ==========================================
# CONFIGURACION Y PARAMETROS
# ==========================================
DATA_PATH       = 'data/data_train.csv'
MODEL_SAVE_PATH = 'data/final_model_state.pth'
LOSS_PLOT_PATH  = 'data/loss_curve_final.png'


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
    print(f"Cargando datos desde {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep=';')
    
    # Preprocesamiento: Convertir VERDADERO/FALSO a 1/0
    df = df.replace({'VERDADERO': 1, 'FALSO': 0})

    # Rellenar NaNs en features con 0
    feature_cols = [c for c in df.columns if c != 'prediccion']
    df[feature_cols] = df[feature_cols].fillna(0)

    # 1. Preparacion de Datos y Normalizacion
    
    # Usar todos los datos
    y_full = df['prediccion'].astype('float32').to_numpy()
    X_full = df[feature_cols].to_numpy(dtype='float32')
    
    # Calcular estadisticas de normalizacion
    X_mean = X_full.mean(axis=0, keepdims=True)
    X_std = X_full.std(axis=0, keepdims=True)

    # Aplicar normalizacion
    X_full = (X_full - X_mean) / (X_std + 1e-8)
    
    print(f"Entrenando con {len(X_full)} registros.")

    # 2. Dataset y Loader
    full_ds = MaskedSeqDataset(X_full, y_full, SEQ_LEN)

    # Sampler balanceado
    labels_full = torch.tensor(full_ds.y[full_ds.valid_ends], dtype=torch.float32)
    pos_count = (labels_full == 1).sum().item()
    neg_count = (labels_full == 0).sum().item()
    pos_weight_full = None

    if pos_count > 0 and neg_count > 0:
        pos_weight_full = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=DEVICE)
        weights_full = torch.where(labels_full == 1, 1.0 / pos_count, 1.0 / neg_count)
        sampler_full = torch.utils.data.WeightedRandomSampler(weights_full, num_samples=len(weights_full), replacement=True)
        full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, sampler=sampler_full)
    else:
        full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Modelo y Entrenamiento
    final_model = LSTMBinary(input_size=len(feature_cols), hidden_size=HIDDEN_SIZE, 
                            num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    crit_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    opt_full = torch.optim.Adam(final_model.parameters(), lr=LR)

    epoch_losses = []
    print("Iniciando entrenamiento final...")

    for epoch in range(1, N_EPOCHS + 1):
        final_model.train()
        running_loss = 0.0
        running_count = 0
        for xb, yb in full_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt_full.zero_grad()
            logits = final_model(xb)
            loss = crit_full(logits, yb)
            loss.backward()
            opt_full.step()
            running_loss += loss.item() * yb.size(0)
            running_count += yb.size(0)
        
        avg_loss = running_loss / max(running_count, 1)
        epoch_losses.append(avg_loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{N_EPOCHS} - Loss: {avg_loss:.4f}")

    # 4. Guardar Modelo y Estado
    # Guardamos tambien X_mean y X_std para poder normalizar nuevos datos en prediccion
    checkpoint = {
        'model_state_dict': final_model.state_dict(),
        'X_mean': X_mean,
        'X_std': X_std,
        'feature_cols': feature_cols,
        'config': {
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'seq_len': SEQ_LEN,
            'threshold': DEFAULT_THRESHOLD
        }
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"Modelo y estado guardados en {MODEL_SAVE_PATH}")

    # 5. Grafico de Loss
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.title("Curva de loss (entrenamiento final)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Grafico de loss guardado en {LOSS_PLOT_PATH}")

if __name__ == "__main__":
    main()
