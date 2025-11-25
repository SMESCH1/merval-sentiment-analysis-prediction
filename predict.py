import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import os
from model_utils import MaskedSeqDataset, LSTMBinary

# ==========================================
# CONFIGURACION Y PARAMETROS

INPUT_FILE      = 'data/data_test.csv'       # Archivo con datos para predecir
TRAIN_FILE      = 'data/data_train.csv'      # Archivo con datos para entrenar

MODEL_PATH      = 'data/final_model_state.pth' # Checkpoint del modelo entrenado
OUTPUT_FILE     = 'data/predictions.csv'      # Donde guardar el resultado

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# SCRIPT PRINCIPAL


def main():
    print(f"Usando dispositivo: {DEVICE}")
    
    # 1. Cargar modelo y configuracion
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
    
    print(f"Modelo desde {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Extraer las características del modelo a partir del .pth
    model_state = checkpoint['model_state_dict']
    X_mean = checkpoint['X_mean']
    X_std = checkpoint['X_std']
    feature_cols = checkpoint['feature_cols']
    config = checkpoint['config']
    
    hidden_size = config['hidden_size']
    num_layers = config.get('num_layers', 1)
    dropout = config['dropout']
    seq_len = config['seq_len']
    threshold = config.get('threshold', 0.5)
    bidirectional = config.get('bidirectional', False)

    # 2. Cargar datos de entrada
    print(f"Datos desde {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, sep=';')

    # Agregar ultimas filas de train para warm-up de secuencias.
    # Esto permite que todas las filas del test puedan aprovecharse y predecirse.
    df_data = pd.read_csv(TRAIN_FILE, sep=';')
    df_data = df_data.iloc[-seq_len:]
    df = pd.concat([df_data, df], ignore_index=True)
    
    # Validar que se tengan las columnas necesarias
    if 'prediccion' not in df.columns:
        raise ValueError("El archivo debe contener columna 'prediccion'")
    
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas: {missing_cols}")
       
    X_raw = df[feature_cols].to_numpy(dtype='float32')
    y_raw = df['prediccion'].astype('float32').to_numpy()
    
    # 3. Normalizar con estadisticas del entrenamiento
    X_norm = (X_raw - X_mean) / (X_std + 1e-8)

    # 4. Dataset y modelo
    ds = MaskedSeqDataset(X_norm, y_raw, seq_len)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    model = LSTMBinary(input_size=len(feature_cols), hidden_size=hidden_size, 
                      num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()
    
    print(f"Predicciones a generar: {len(ds)}")
    
    # 5. Generar predicciones
    probs_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            
    probs_all = np.concatenate(probs_list)
    preds_all = (probs_all > threshold).astype(int)
    
    # 6. Guardar resultados
    valid_ends = ds.valid_ends
    

    # Preparo df con predicciones
    df_out = df.copy()
    df_out['pred_prob'] = np.nan
    df_out['pred_class'] = np.nan
    df_out.loc[valid_ends, 'pred_prob'] = probs_all
    df_out.loc[valid_ends, 'pred_class'] = preds_all
    
    # Eliminar filas de warm-up del output final
    df_out = df_out.iloc[len(df_data):]
    
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"Resultados guardados en {OUTPUT_FILE}")
    print(f"Total predicciones: {len(probs_all)}")
    
    # 7. Matriz de confusion
    if 'prediccion' in df_out.columns:
        valid_comparison = df_out.dropna(subset=['pred_class', 'prediccion'])
        
        if len(valid_comparison) > 0:
            y_true = valid_comparison['prediccion'].astype(int)
            y_pred = valid_comparison['pred_class'].astype(int)
            
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            print("\n" + "="*30)
            print("MATRIZ DE CONFUSION")
            print("="*30)
            print(f"TP: {tp}  FP: {fp}")
            print(f"FN: {fn}  TN: {tn}")
            print("="*30)

if __name__ == "__main__":
    main()
