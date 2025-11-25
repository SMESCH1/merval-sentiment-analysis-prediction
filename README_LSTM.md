# Predicción de Precios de Acciones con LSTM

---

## Resumen Ejecutivo

Este proyecto implementa un clasificador binario basado en redes LSTM (Long Short-Term Memory) para predecir la dirección del movimiento de precios de acciones. El sistema utiliza datos históricos de actividad en redes sociales, indicadores macroeconómicos y actividad bursátil para generar predicciones binarias (sube/baja).

**Tecnologías utilizadas:**
- Python 3.x
- PyTorch (Deep Learning Framework)
- Optuna (Optimización de Hiperparámetros)
- Scikit-learn (Métricas y Validación)

---

## Introducción

### Motivación

La predicción de mercados financieros es un problema complejo debido a la naturaleza no lineal y estocástica de las series temporales bursátiles. Este proyecto explora el uso de redes neuronales recurrentes (específicamente LSTM) para capturar patrones temporales en datos financieros y generar predicciones útiles.

### Objetivos

1. Implementar un clasificador binario basado en LSTM
2. Optimizar hiperparámetros usando búsqueda bayesiana (Optuna)
3. Evaluar el modelo usando cross-validation temporal
4. Implementar técnicas de regularización (early stopping, dropout)
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
- Filtra secuencias con labels NaN
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

Implementa parada temprana basada en AUC de validación:
- Monitorea mejora en validación cada epoch
- Para si no hay mejora por `patience` epochs
- Usa período de warmup (min_epochs) para estabilidad

#### 4. Cross-Validation Temporal (`rolling_splits`)

Genera splits respetando el orden temporal:
```
Split 1: [Train: 0-650]    [Val: 650-750]
Split 2: [Train: 100-750]  [Val: 750-850]
Split 3: [Train: 200-850]  [Val: 850-950]
...
```

---

## Metodología

### 1. Preprocesamiento de Datos

**Features utilizados:**
- `REDDIT`: Actividad en redes sociales
- `R USD`: Variación del dólar
- `R MERVAL`: Variación del índice Merval
- `actividad`: Indicador binario de actividad bursátil

**Normalización:**
```python
X_normalized = (X - X_train.mean()) / (X_train.std() + epsilon)
```

**Importante:** Solo se usan estadísticas del conjunto de entrenamiento para evitar data leakage.

### 2. Optimización de Hiperparámetros

Se utiliza Optuna con pruner mediano para optimizar:

| Hiperparámetro | Rango | Tipo |
|----------------|-------|------|
| seq_len | 70-100 | Entero (step=7) |
| batch_size | 16-64 | Entero (step=16) |
| hidden_size | 4-42 | Entero (step=4) |
| dropout | 0.0-0.5 | Continuo |
| learning_rate | 1e-4 - 1e-2 | Log-uniforme |
| n_epochs | 50-100 | Entero (step=25) |
| bidirectional | {True, False} | Categórico |

**Función objetivo:** AUC promedio en validación cruzada

### 3. Validación Cruzada

**Configuración:**
- Tipo: Rolling Window Cross-Validation
- Train size: 650 muestras
- Validation size: 100 muestras
- Número de folds: Variable (depende de datos disponibles)

**Curva ROC:**  
Las predicciones de todos los folds se concatenan para generar una curva ROC global. Esto es válido porque cada predicción es out-of-sample (el modelo no vio esos datos durante entrenamiento).

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

### Paso 1: Optimización (Opcional)

```bash
python optuna_search.py
```

Este script ejecuta 200 trials de búsqueda bayesiana y genera:
- `data/best_params.csv`: Mejores hiperparámetros encontrados
- `data/optuna_study.db`: Base de datos con historial completo
- Visualizaciones de convergencia e importancia

### Paso 2: Validación Cruzada

```bash
python training.py
```

Ejecuta cross-validation temporal y reporta:
- AUC por fold
- Estadísticas de early stopping
- Recomendación de epochs óptimos
- Curva ROC global
- Análisis de umbrales de decisión

### Paso 3: Entrenamiento Final

```bash
python train_final.py
```

Entrena modelo con todos los datos usando hiperparámetros optimizados. El modelo y estadísticas se guardan en `data/final_model_state.pth`.

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

**Cross-Validation:**
- AUC promedio: [Completar con tus resultados]
- Desviación estándar: [Completar]

**Conjunto de Prueba:**
- AUC: [Completar]
- Accuracy: [Completar]
- F1-Score: [Completar]

**Matriz de Confusión (Test):**
```
        Predicho
        Baja  Sube
Real Baja  TN    FP
     Sube  FN    TP
```

### Análisis de Hiperparámetros

**SEQ_LEN (Ventana Temporal):**
- Valores bajos (<50): Poco contexto temporal
- Valores altos (>90): Riesgo de overfitting
- Óptimo encontrado: 90 días

**BIDIRECTIONAL:**
- False (unidireccional) resultó óptimo
- Bidireccional aumenta parámetros sin mejora significativa

---

## Limitaciones y Trabajo Futuro

### Limitaciones

1. **Tamaño del dataset:** 1014 muestras pueden ser insuficientes para LSTM profundas
2. **Features limitados:** Solo 4 variables de entrada
3. **Predicción binaria:** No captura magnitud del movimiento
4. **Datos desbalanceados:** Posibles NaN en períodos no bursátiles

### Mejoras Propuestas

1. **Aumentar datos:** Incorporar más fuentes de información
2. **Feature engineering:** Indicadores técnicos adicionales
3. **Ensemble methods:** Combinar múltiples modelos
4. **Atención mechanism:** Implementar attention layers
5. **Predicción multi-clase:** Categorizar magnitud del cambio

---

## Conclusiones

Este proyecto demuestra la viabilidad de usar redes LSTM para predicción de mercados financieros. Las principales conclusiones son:

1. **Early stopping es crucial:** Evita overfitting y reduce tiempo de entrenamiento
2. **Optimización de hiperparámetros:** Mejora significativa vs. parámetros por defecto
3. **Cross-validation temporal:** Esencial para evaluar correctamente modelos en series temporales
4. **AUC como métrica:** Más robusta que accuracy para problemas de clasificación

El sistema desarrollado proporciona una base sólida para investigación futura en predicción financiera usando deep learning.

---

## Referencias

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

2. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of machine learning research, 13(2).

3. Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. In KDD.

4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

---

## Anexos

### A. Instalación

```bash
pip install torch numpy pandas scikit-learn optuna matplotlib
```

### B. Ejecución Completa

```bash
# 1. Optimizar hiperparámetros
python optuna_search.py

# 2. Validar con CV
python training.py

# 3. Entrenar modelo final
python train_final.py

# 4. Generar predicciones
python predict.py
```

### C. Estructura del Checkpoint

```python
{
    'model_state_dict': torch.nn.Module.state_dict(),
    'config': {
        'hidden_size': int,
        'num_layers': int,
        'dropout': float,
        'seq_len': int,
        'bidirectional': bool
    },
    'X_mean': np.ndarray,
    'X_std': np.ndarray,
    'feature_cols': list[str]
}
```

---

**Código fuente disponible en:** [URL del repositorio]

**Contacto:** [Tu email universitario]
