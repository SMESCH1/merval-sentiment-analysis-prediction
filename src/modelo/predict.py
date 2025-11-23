import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import os
from model_utils import InferenceDataset, LSTMBinary

# ==========================================
# CONFIGURACION Y PARAMETROS
# ==========================================
INPUT_FILE      = 'data/data_test.csv'       # Archivo con datos para predecir
MODEL_PATH      = 'data/final_model_state.pth' # Checkpoint del modelo entrenado
OUTPUT_FILE     = 'data/predictions.csv'      # Donde guardar el resultado

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    print(f"Usando dispositivo: {DEVICE}")
    
    # 1. Cargar Checkpoint
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontro el modelo en {MODEL_PATH}. Ejecuta train_final.py primero.")
    
    print(f"Cargando modelo desde {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Recuperar configuracion y stats
    model_state = checkpoint['model_state_dict']
    X_mean      = checkpoint['X_mean']
    X_std       = checkpoint['X_std']
    feature_cols= checkpoint['feature_cols']
    config      = checkpoint['config']
    
    hidden_size = config['hidden_size']
    num_layers = config.get('num_layers', 1)  # Default a 1 si no existe (compatibilidad)
    dropout     = config['dropout']
    seq_len     = config['seq_len']
    threshold   = config.get('threshold', 0.5)
    
    print(f"Configuracion recuperada: SEQ_LEN={seq_len}, HIDDEN={hidden_size}, LAYERS={num_layers}, THRESHOLD={threshold}")

    # 2. Cargar Datos de Entrada
    print(f"Cargando datos de entrada desde {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, sep=';')
    # Preprocesamiento: Convertir VERDADERO/FALSO a 1/0
    df = df.replace({'VERDADERO': 1, 'FALSO': 0})
    
    # Rellenar NaNs en features con 0
    df = df.fillna(0)
    
    # Verificar columnas
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas en el archivo de entrada: {missing_cols}")
    
    # Seleccionar features y normalizar
    X_raw = df[feature_cols].to_numpy(dtype='float32')
    
    # Aplicar la MISMA normalizacion que en el entrenamiento
    X_norm = (X_raw - X_mean) / (X_std + 1e-8)
    
    print(f"Datos normalizados. Dimensiones: {X_norm.shape}")

    # 3. Preparar Dataset de Inferencia
    # InferenceDataset genera ventanas deslizantes para todo punto valido
    ds = InferenceDataset(X_norm, seq_len)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # 4. Cargar Modelo
    model = LSTMBinary(input_size=len(feature_cols), hidden_size=hidden_size, 
                      num_layers=num_layers, dropout=dropout)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()
    
    # 5. Predecir
    print("Generando predicciones...")
    probs_list = []
    
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            
    if len(probs_list) == 0:
        print("No se generaron predicciones (quizas la serie es muy corta).")
        return

    probs_all = np.concatenate(probs_list)
    preds_all = (probs_all > threshold).astype(int)
    
    # 6. Guardar Resultados
    # InferenceDataset.valid_ends nos dice a que indice del DF original corresponde cada prediccion
    valid_ends = ds.valid_ends
    
    # Creamos un DF con las predicciones alineadas
    # Inicializamos con NaN
    df_out = df.copy()
    df_out['pred_prob'] = np.nan
    df_out['pred_class'] = np.nan
    
    # Asignamos valores en las posiciones validas
    df_out.loc[valid_ends, 'pred_prob'] = probs_all
    df_out.loc[valid_ends, 'pred_class'] = preds_all
    
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"Predicciones guardadas en {OUTPUT_FILE}")
    print(f"Total predicciones generadas: {len(probs_all)}")

if __name__ == "__main__":
    main()
