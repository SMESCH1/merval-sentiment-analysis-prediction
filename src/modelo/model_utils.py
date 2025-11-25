import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class MaskedSeqDataset(torch.utils.data.Dataset):
    """
    Dataset para secuencias temporales con labels posiblesmente NaN.
    La idea es evitar que el modelo se entrene con casos en los que el último día de 
    la secuencia es sea uno previo a un feriado/ fin de semana (y por lo tanto, un día
    cuyo valor en el atributo 'prediccion' no es válido y tiene valor NaN).
    """
    
    def __init__(self, X, y, seq_len):
        """
        X: (T, n_features)
        y: (T,) con posibles NaN
        seq_len: longitud de la ventana temporal
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len
        T = len(X)

        # Registro los índices validos que pueden ser el ultimo dia de una secuencia
        # Lo que significa que (i >= seq_len-1) y y[i] no es NaN
        
        finales_validos = []
        for i in range(seq_len - 1, T):
            if not np.isnan(y[i]):
                finales_validos.append(i)

        self.finales_validos = np.array(finales_validos, dtype=int)
        # Mantener compatibilidad con codigo anterior
        self.valid_ends = self.finales_validos

    def __len__(self):
        return len(self.finales_validos)

    def __getitem__(self, idx):
        end_idx = self.finales_validos[idx]        # indice del label
        start_idx = end_idx - self.seq_len + 1     # inicio de la ventana

        x_seq = self.X[start_idx:end_idx + 1]      # (seq_len, n_features)
        y_lbl = self.y[end_idx]                    

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_lbl, dtype=torch.float32),
        )


def rolling_splits(N, train_size, val_size):
    """
    Genera listas con elementos=(train_start, train_end, val_start, val_end).
    Indican los índices relevantes de los folds dentro del DataFrame.
    """
    splits = []
    train_start = 0

    while True:
        train_end = train_start + train_size
        val_start = train_end
        val_end = val_start + val_size
        if val_end > N:
            break
        splits.append((train_start, train_end, val_start, val_end))
        train_start += val_size  # avanzamos un bloque de validacion

    return splits

class LSTMBinary(nn.Module):
    """
    Modelo LSTM binario para clasificacion de secuencias.
    Incluye una LSTM y una capa densa lineal para ajustar el output a un escalar.
    Soporta arquitectura bidireccional.
    """
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        
        # Si es bidireccional, el hidden state es el doble
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            # Concatenar forward y backward del ultimo layer
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch, hidden_size * 2)
        else:
            last_hidden = h_n[-1]  # (batch, hidden_size)
        
        dropped = self.dropout(last_hidden)
        logit = self.fc(dropped)       # (batch, 1)
        return logit.squeeze(-1)       # (batch,)


def train_fold(model, train_loader, val_loader, n_epochs=5, 
               lr=1e-3, device='cpu', threshold=0.5, patience=10, min_delta=0.0001):
    
    """
    Entrena un fold con early stopping basado en AUC de validación.
    Loss utilizada: BCEWithLogitsLoss, que implica que el output del modelo es un logit.
    Optimizador: Adam
    Retorna: (mejor_auc, mejor_epoch)
    
    Args:
        patience: Número de epochs sin mejora antes de detener
        min_delta: Mínima mejora considerada como progreso
    """
    
    model.to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Variables para early stopping
    best_auc = -1.0  # Mejor AUC. Iniciar en -1 para que cualquier valor sea mejor
    best_epoch = 1 # Mejor epoch
    epochs_no_improve = 0 # Cantidad de epochs sin mejora
    min_epochs = max(10, patience)  # Entrenar al menos 10 epochs o patience, lo que sea mayor
    
    for epoch in range(n_epochs):
        # Entrenamiento
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        # Evaluación en validación
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs)
                val_labels.append(yb.numpy())
        
        if len(val_probs) > 0:
            val_probs = np.concatenate(val_probs)
            val_labels = np.concatenate(val_labels)
            
            if len(np.unique(val_labels)) > 1:
                current_auc = roc_auc_score(val_labels, val_probs)
                
                # Early stopping check
                if current_auc > best_auc + min_delta:
                    best_auc = current_auc
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                # Se permite entrenar al menos min_epochs epochs 
                if epoch >= min_epochs and epochs_no_improve >= patience:
                    # print(f"  Early stopping en epoch {epoch + 1}. Mejor AUC: {best_auc:.4f} (epoch {best_epoch})")
                    break
    
    # Si best_auc sigue en -1, no hubo ninguna mejora válida
    if best_auc < 0:
        best_auc = 0.0
    
    return best_auc, best_epoch
