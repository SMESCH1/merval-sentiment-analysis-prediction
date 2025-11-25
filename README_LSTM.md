# Predicción de Precios de Acciones con LSTM

---

## Resumen Ejecutivo

Este proyecto implementa un clasificador binario basado en redes LSTM (Long Short-Term Memory) para predecir la dirección del movimiento de precios de acciones. El sistema utiliza como fuentes datos de REDDIT, retornos de MERVAL y dólar, así como un booleano que indica si hubo actividad bursátil en dicho día. 
La predicción es binaria, es decir, se predice si el precio del índice MERVAL sube o baja al día siguiente.

---

## Introducción

### Motivación

La predicción de mercados financieros es un problema complejo debido a la naturaleza no lineal y estocástica de las series temporales bursátiles. Este proyecto explora el uso de redes neuronales recurrentes (específicamente LSTM) para capturar patrones temporales en datos financieros y generar predicciones útiles.

### Objetivos

1. Implementar un clasificador binario basado en LSTM
2. Optimizar hiperparámetros usando búsqueda bayesiana (Optuna)
3. Evaluar el modelo usando cross-validation temporal
5. Analizar el rendimiento mediante métricas estándar (AUC, F1-Score, etc.)

---

## Arquitectura del Sistema

### Estructura de Archivos

```
stock_price_propio/
├── data/
│   ├── data_train.csv          # Conjunto de entrenamiento
│   ├── data_test.csv           # Conjunto de prueba
│   ├── final_model_state.pth   # Modelo entrenado
│   └── optuna_study.db         # Base de datos de optimización
├── model_utils.py              # Clases y funciones auxiliares
├── optuna_search.py            # Optimización de hiperparámetros
├── training.py                 # Entrenamiento con validación cruzada
├── train_final.py              # Entrenamiento del modelo final
├── predict.py                  # Generación de predicciones
└── plot_roc.py                 # Visualización de resultados
```

### Componentes Principales

#### 1. Dataset (`MaskedSeqDataset`)

Implementa un dataset personalizado de PyTorch que:
- Crea ventanas deslizantes de longitud `seq_len`
- Evita generar secuencias cuyo último día sea anterior a un día sin actividad bursátil. Esto es para evitar generar predicciones para días en los que el índice MERVAL no cambia.
- Retorna tuplas (secuencia, label)

#### 2. Modelo LSTM (`LSTMBinary`)

Arquitectura de red neuronal:
```
Input (seq_len, n_features) 
    ↓
LSTM (num_layers, hidden_size, bidirectional)
    ↓
Dropout (p = dropout)
    ↓
Linear (hidden_size × direction_multiplier → 1)
    ↓
Output (logit)
```

#### 3. Early Stopping

Implementado en script de training, se monitorea AUC de validación:
- Monitorea mejora en validación en cada epoch
- Para si no hay mejora por `patience` epochs
- Usa período de warmup (min_epochs) para estabilidad

#### 4. Cross-Validation Temporal (`rolling_splits`)

Genera los índices para realizar folds respetando el orden temporal:
```
Fold 1: [Train: 0-650]    [Val: 650-750]
Fold 2: [Train: 100-750]  [Val: 750-850]
Fold 3: [Train: 200-850]  [Val: 850-950]
...
```
La intención es entrenar y evaluar un modelo con cada fold con la información disponible en cada uno.

---

## Metodología

### 1. Preprocesamiento de Datos

**Features utilizados:**
- `REDDIT`: Actividad en redes sociales
- `R USD`: Retorno logarítmico del dólar
- `R MERVAL`: Retorno logarítmico del índice MERVAL
- `actividad`: Indicador binario de actividad bursátil

**Normalización:**
Se normalizan las variables numéricas usando estadísticas del conjunto de entrenamiento para evitar data leakage.
El modelo final incluye esta información para poder aplicársele al test set.
```python
X_normalized = (X - X_train.mean()) / (X_train.std() + epsilon)
```

### 2. Optimización de Hiperparámetros

Se utiliza Optuna con pruner para optimizar:

| Hiperparámetro | Rango | Tipo |
|----------------|-------|------|
| seq_len | 70-100 | Entero (step=7) |
| batch_size | 16-64 | Entero (step=16) |
| hidden_size | 4-42 | Entero (step=4) |
| dropout | 0.0-0.5 | Continuo |
| learning_rate | 1e-4 - 1e-2 | Log-uniforme |
| n_epochs | 50-100 | Entero (step=25) |
| bidirectional | {True, False} | Categórico |

**Función objetivo:** AUC promedio en validación cruzada.  Esto es, se realizan distintos folds y se promedia el AUC obtenido de cada uno.

### 3. Validación Cruzada

**Configuración:**
- Tipo: Rolling Window Cross-Validation
- Train size: 650 muestras
- Validation size: 100 muestras

**Curva ROC:**  
Las predicciones de todos los folds se concatenan para generar una curva ROC global. Esto es válido porque cada predicción es out-of-sample (el modelo no vio esos datos durante el entrenamiento).

### 4. Métricas de Evaluación

| Métrica | Descripción | Uso |
|---------|-------------|-----|
| **AUC** | Área bajo curva ROC | Métrica principal |
| **Accuracy** | (TP+TN)/Total | Rendimiento general |
| **Precision** | TP/(TP+FP) | Evitar falsas alarmas |
| **Recall** | TP/(TP+FN) | Detectar todos los positivos |
| **F1-Score** | Media armónica P/R | Balance precision/recall |

---

## Implementación

### Paso 1: Búsqueda de Hiperparámetros con Optuna.

```bash
python optuna_search.py
```

Este script ejecuta 200 trials de búsqueda bayesiana y genera:
- `data/optuna_study.db`: Base de datos con historial completo
- Visualizaciones de convergencia e importancia

### Paso 2: Validación Cruzada

```bash
python training.py
```
Se realiza una validación cruzada con los hiperparámetros optimizados. 
Ejecuta cross-validation temporal y reporta:
- AUC por fold
- Estadísticas de early stopping. En particular se analiza el número de epochs que se usaron para entrenar cada fold antes de que el early stopping se activara. Resulta importante porque con el entrenamiento final del modelo se deben utilizar todos los datos disponibles, lo que imposibilita destinar los últimos datos del training set en generar un conjunto de validación.
- Curva ROC global
- Análisis de umbrales de decisión

### Paso 3: Entrenamiento Final

```bash
python train_final.py
```

Entrena modelo con todos los datos disponibles en el training set usando hiperparámetros optimizados. El modelo y estadísticas se guardan en `data/final_model_state.pth`.

### Paso 4: Predicción

```bash
python predict.py
```

Genera predicciones en datos de prueba y calcula métricas finales.

---

## Resultados

### Configuración Óptima Encontrada

```python
SEQ_LEN = 90           # Ventana temporal
HIDDEN_SIZE = 28       # Neuronas LSTM
DROPOUT = 0.46         # Tasa de dropout
LEARNING_RATE = 0.00057
BIDIRECTIONAL = False  # LSTM unidireccional
```

### Rendimiento

**Matriz de Confusión (Test):**
```
        Predicho
        Baja  Sube
Real Baja  15    9
     Sube  0    2
```
---
