import numpy as np
import torch
import torch.nn as nn

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
    """
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # (batch, hidden_size)
        dropped = self.dropout(last_hidden)
        logit = self.fc(dropped)       # (batch, 1)
        return logit.squeeze(-1)       # (batch,)


def train_fold(model, train_loader, val_loader, n_epochs=5, 
               lr=1e-3, device='cpu', threshold=0.5):
    
    """
    Entrena un fold.
    Loss utilizada: BCEWithLogitsLoss, que implica que el output del modelo es un logit.
    Optimizador: Adam
    """
    
    model.to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

    # evaluacion: accuracy y precision
    model.eval()
    correct, total = 0, 0
    tp, fp = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()
            tp += ((preds == 1) & (yb == 1)).sum().item()
            fp += ((preds == 1) & (yb == 0)).sum().item()

    acc = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return acc, precision
