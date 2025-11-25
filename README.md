# Proyecto: Predicción del valor del índice Merval utilizando un LSTM con señales de sentimiento de Reddit

---
## Resumen
Este proyecto realiza análisis de sentimiento sobre datos de subreddits de finanzas de Argentina e implementa un clasificador binario basado en redes LSTM (Long Short-Term Memory) para predecir la dirección del movimiento de precios de acciones. El sistema utiliza como fuentes datos de REDDIT, retornos de MERVAL y dólar, así como un booleano que indica si hubo actividad bursátil en dicho día. 
La predicción es binaria, es decir, se predice si el precio del índice MERVAL sube o baja al día siguiente.

---
## PPT Final
[Presentación de diapositivas del TP](https://docs.google.com/presentation/d/e/2PACX-1vQxqObrHd2g0kOkwWzr-udYWEFjN8OG7mZkR9wlRs6HeNHhtzN6N1P0gilSl_TVJTZczMcXIHu1_fmx/pub?start=false&loop=false&delayms=3000)

### Motivación

La predicción de mercados financieros es un problema complejo debido a la naturaleza no lineal y estocástica de las series temporales bursátiles. Este proyecto explora el uso de redes neuronales recurrentes (específicamente LSTM) para capturar patrones temporales en datos financieros y generar predicciones útiles.

### Objetivos

1. Implementar un clasificador binario basado en LSTM
2. Optimizar hiperparámetros usando búsqueda bayesiana (Optuna)
3. Evaluar el modelo usando cross-validation temporal
5. Analizar el rendimiento mediante métricas estándar (AUC, F1-Score, etc.)

---

## Arquitectura del Proyecto

1. **Ingesta de Datos**
   - Scraping de Reddit (comentarios y posts).
   - Descarga de datos de mercado (usando `yfinance`).
   - Datos de dólar blue

2. **Procesamiento**
   - Limpieza de texto (stopwords, emojis, jerga argentina).
   - Análisis de sentimiento usando `pysentimiento` (modelo RoberTuito).
   - Agregación temporal 

3. **Modelado y Predicción**
   - LSTM para predicción binaria (sube/baja).
   - Validación walk-forward para series temporales.
   - Comparación de performance con/sin señales de texto.

---

## Estructura de archivos

```
project-root/
│
├── data/
│   ├── preprocesada/          # Datos unificados y limpios (JSONL)
│   ├── procesada/             # Datos con análisis de sentimiento (JSONL)
│   ├── raw/                   # Datos crudos (CSV de dólar, etc.)
│   └── historica*/            # Datos históricos scrapeados (JSON/CSV por mes)
│
├── src/
│   ├── scraping/
│   │   ├── scraper_historico.py  # Scraper histórico de Reddit
│   │   └── reddit_config.py      # Configuración de Reddit API
│   │
│   ├── processing/
│   │   ├── etl_pipeline.py       # Pipeline ETL principal
│   │   ├── sentiment_pipeline.py  # Pipeline de análisis de sentimiento
│   │   ├── sentiment_analyzer.py # Analizador de sentimiento (pysentimiento)
│   │   ├── run_etl.py            # Script CLI para ETL
│   │   ├── run_sentiment.py      # Script CLI para sentiment analysis
│   │   ├── schema.py             # Esquemas de datos (Pydantic)
│   │   ├── loaders.py            # Cargadores de datos
│   │   ├── data_transformers.py  # Transformadores de datos
│   │   └── cleaners.py           # Limpieza de texto
│   │
│   └── modelo/
│       ├── model_utils.py        # Utilidades del modelo (MaskedSeqDataset, LSTMBinary)
│       ├── training.py           # Entrenamiento con Cross-Validation
│       ├── train_final.py       # Entrenamiento final con todos los datos
│       ├── optuna_search.py     # Optimización de hiperparámetros
│       ├── predict.py           # Generar predicciones
│       ├── plot_roc.py          # Visualizar curva ROC
│       └── data/                # Datos del modelo (CSV, modelos entrenados, etc.)
│           ├── data_train.csv
│           ├── data_test.csv
│           ├── final_model_state.pth
│           ├── optuna_study.db
│           └── ...
│
├── notebooks/
│   ├── generar_datasets.ipynb      # Generación de datasets de entrenamiento y test
│   └── analisis_lstm_completo.ipynb # Análisis completo del modelo
│
├── docs/
│   ├── bibliografía.md
│   └── documentacion_proyecto.docx
│
├── logs/                      # Logs de ejecución
├── requirements.txt           # Dependencias Python
└── README.md                  # Este archivo
```

---

## Flujo de Trabajo

### Prerrequisitos
- Python 3.11+
- Cuenta de Reddit con API credentials

### 1. Configurar Credenciales de Reddit

Crear archivo `.env` con las credenciales de Reddit:
```
client_id=tu_client_id
client_secret=tu_client_secret
reddit_username=tu_usuario
reddit_password=tu_password
```

### 2. Scraping Histórico de Reddit 

Script para obtener datos históricos de Reddit y aumentar el tamaño del dataset:

```bash
# Scraping histórico con filtro de keywords (solo posts financieros)
python src/scraping/scraper_historico.py \
    --start-date 2020-01-01 \
    --end-date 2024-12-31

```

### 3. Pipeline ETL

```bash
# Procesar datos de reddit
python src/processing/run_etl.py \
    --reddit-only \
    --output-dir data/preprocesada \
    --format jsonl \
    --verbose

```
### 4. Análisis de Sentimiento

Aplica análisis de sentimiento usando `pysentimiento` (modelo RoberTuito) a los datos preprocesados

```bash

# Ejecutar análisis de sentimiento
python src/processing/run_sentiment.py \
    data/preprocesada/unified_data_2025.jsonl \
    --output data/procesada/reddit_unified_2025_with_sentiment.jsonl \
    --batch-size 32 \
    --verbose
```
### 5. Preparar Dataset para el Modelo

Los datasets de entrenamiento y test se generan usando el notebook `notebooks/generar_datasets.ipynb`. Este notebook:

1. Carga datos de Reddit con análisis de sentimiento
2. Extrae la probabilidad positiva de sentimiento (`pos_prob_mean`) agrupada por día
3. Descarga datos de MERVAL desde yfinance y calcula retorno logarítmico
4. Carga datos del dólar desde CSV y calcula retorno logarítmico
5. Crea la columna `prediccion` (target): 1 si MERVAL sube al día siguiente, 0 si baja
6. Guarda los datasets en formato CSV con separador `;`

**Columnas del dataset**:
- `pos_prob_mean`: Media de probabilidad positiva de sentimiento por día
- `retorno_log_merval`: Retorno logarítmico del MERVAL
- `retorno_log_dolar`: Retorno logarítmico del dólar
- `prediccion`: Target (1 si MERVAL sube al día siguiente, 0 si baja, NaN para días no hábiles)

### 6. Entrenar el Modelo LSTM

El modelo LSTM se encuentra en `src/modelo/`. El proceso de entrenamiento sigue un flujo específico que incluye optimización de hiperparámetros, validación cruzada, y entrenamiento final.

#### Componentes Principales del Modelo

**1. Dataset (`MaskedSeqDataset`)**
- Crea ventanas deslizantes de longitud `seq_len` (típicamente 90 días)
- Evita generar secuencias cuyo último día sea anterior a un día sin actividad bursátil (fines de semana, feriados)
- Filtra automáticamente días donde el target (`prediccion`) tiene valor NaN
- Retorna tuplas (secuencia, label) para entrenamiento

**2. Modelo LSTM (`LSTMBinary`)**
Arquitectura de red neuronal:
```
Input (seq_len, n_features) 
    ↓
LSTM (num_layers=1, hidden_size, bidirectional=False)
    ↓
Dropout (p = dropout)
    ↓
Linear (hidden_size → 1)
    ↓
Output (logit) → Sigmoid → Probabilidad [0,1]
```

**3. Early Stopping**
- Monitorea AUC de validación en cada epoch
- Detiene el entrenamiento si no hay mejora por `patience` epochs (70 por defecto)
- Usa período de warmup (min_epochs) para estabilidad

**4. Cross-Validation Temporal (`rolling_splits`)**
Genera folds respetando el orden temporal:
```
Fold 1: [Train: 0-650]    [Val: 650-750]
Fold 2: [Train: 100-750]  [Val: 750-850]
Fold 3: [Train: 200-850]  [Val: 850-950]
...
```
Cada fold entrena y evalúa un modelo con la información disponible hasta ese momento.

#### 6.1. Preparar datos para el modelo

**Preprocesamiento:**
- Features utilizados: `pos_prob_mean`, `retorno_log_merval`, `retorno_log_dolar`
- Normalización: Se normalizan las variables usando estadísticas del conjunto de entrenamiento para evitar data leakage
- Manejo de NaN: Features con NaN se rellenan con 0, targets con NaN se filtran automáticamente

#### 6.2. Optimización de Hiperparámetros (Optuna)

**Paso 1: Búsqueda de Hiperparámetros**

```bash
cd src/modelo
python optuna_search.py [n_trials]
```

Este script ejecuta búsqueda bayesiana (por defecto 200 trials) y optimiza:

| Hiperparámetro | Rango | Tipo |
|----------------|-------|------|
| seq_len | 70-100 | Entero (step=7) |
| batch_size | 16-64 | Entero (step=16) |
| hidden_size | 4-42 | Entero (step=4) |
| dropout | 0.0-0.5 | Continuo |
| learning_rate | 1e-4 - 1e-2 | Log-uniforme |
| n_epochs | 50-100 | Entero (step=25) |
| bidirectional | {True, False} | Categórico |

**Función objetivo:** AUC promedio en validación cruzada temporal

**Archivos generados:**
- `src/modelo/data/optuna_study.db`: Base de datos SQLite con historial completo
- `src/modelo/data/optuna_trials.csv`: Historial de todos los trials
- `src/modelo/data/best_params.csv`: Mejores hiperparámetros encontrados
- `src/modelo/data/param_importance.png`: Importancia de cada parámetro
- `src/modelo/data/optimization_history.png`: Historial de convergencia

**Ejecutar en background:**
```bash
cd src/modelo
nohup python optuna_search.py 200 > ../../logs/optuna.log 2>&1 &
tail -f ../../logs/optuna.log
```

#### 6.3. Entrenamiento con Cross-Validation

**Paso 2: Validación Cruzada Temporal**

```bash
cd src/modelo
python training.py
```

Se realiza validación cruzada con los hiperparámetros optimizados. El script ejecuta:

**Configuración:**
- Tipo: Rolling Window Cross-Validation
- Train size: 650 muestras
- Validation size: 100 muestras
- Se desplaza temporalmente para simular predicciones en el futuro

**Reportes generados:**
- AUC por fold
- Estadísticas de early stopping: número de epochs usados en cada fold antes de que se active el early stopping (importante para determinar epochs en entrenamiento final)
- Curva ROC global: predicciones de todos los folds concatenadas (válido porque cada predicción es out-of-sample)
- Análisis de umbrales de decisión: evaluación de diferentes umbrales (0.3, 0.4, 0.5, 0.6, 0.7)

**Archivos generados:**
- `src/modelo/data/val_predictions.csv`: Predicciones de validación (probabilidades y labels)
- `src/modelo/data/roc_curve.png`: Curva ROC global

#### 6.4. Entrenamiento Final

**Paso 3: Entrenamiento del Modelo Final**

```bash
cd src/modelo
python train_final.py
```

Entrena el modelo con todos los datos disponibles en el training set usando los hiperparámetros optimizados. 

**Características:**
- Utiliza el número de epochs recomendado por las estadísticas de early stopping de la validación cruzada
- Normaliza usando estadísticas de todo el conjunto de entrenamiento
- Guarda el modelo, estadísticas de normalización, y configuración para uso posterior

**Archivos generados:**
- `src/modelo/data/final_model_state.pth`: Modelo entrenado con metadatos (configuración, estadísticas de normalización, feature columns)
- `src/modelo/data/loss_curve_final.png`: Curva de pérdida durante el entrenamiento

#### 6.5. Generar Predicciones

**Paso 4: Predicción en Conjunto de Test**

```bash
cd src/modelo
python predict.py
```

Genera predicciones en datos de prueba y calcula métricas finales.

**Proceso:**
- Carga el modelo entrenado y sus metadatos
- Agrega las últimas filas del training set para warm-up de secuencias (permite predecir todas las filas del test)
- Aplica la misma normalización usada en entrenamiento
- Genera probabilidades y predicciones binarias (usando umbral configurado)

**Archivos generados:**
- `src/modelo/data/predictions.csv`: Predicciones con probabilidades (`pred_prob`) y clases (`pred_class`)
- Muestra matriz de confusión si hay labels disponibles en el test set

#### 6.6. Visualizar Resultados

```bash
cd src/modelo
python plot_roc.py --input src/modelo/data/val_predictions.csv
```

Genera visualización de la curva ROC desde predicciones guardadas.

**Métricas de Evaluación:**

| Métrica | Descripción | Uso |
|---------|-------------|-----|
| **AUC** | Área bajo curva ROC | Métrica principal |
| **Accuracy** | (TP+TN)/Total | Rendimiento general |
| **Precision** | TP/(TP+FP) | Evitar falsas alarmas |
| **Recall** | TP/(TP+FN) | Detectar todos los positivos |
| **F1-Score** | Media armónica P/R | Balance precision/recall |

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
