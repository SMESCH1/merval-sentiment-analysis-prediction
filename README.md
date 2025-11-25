# Proyecto: Predicción de Precios de Activos Financieros en Argentina usando Deep Learning y Datos Sociales

---

## Índice

1. [Descripción general]()



---

[Presentación de diapositivas del TP](https://docs.google.com/presentation/d/1htcqRN_dIlC_S9j6hK2fxhk_o9ao8IVpJe8fz_M0aqQ/edit?slide=id.g39ade7ba14d_0_134#slide=id.g39ade7ba14d_0_134), etapa 1

## Descripción General

Este proyecto busca predecir precios de activos financieros en Argentina combinando:
- **Datos de mercado** (acciones, bonos, dólar, etc.).
- **Noticias económicas y financieras** de medios argentinos.
- **Discusiones en Reddit** (subreddits relacionados con economía y finanzas).

El objetivo es analizar si las señales sociales (sentimiento y volumen de menciones) aportan valor predictivo a los modelos clásicos de series temporales.

Todo el flujo se ejecuta en **infraestructura local, gratuita y open source**, sin depender de servicios pagos en la nube.

---

## Arquitectura del Proyecto

1. **Ingesta de Datos**
   - Scraping de Reddit (comentarios y posts).
   - Scraping de noticias de economía y finanzas de Argentina (deseable, no implementada; se utiliza una dbs de noticias que cuenta con SA)
   - Descarga de datos de mercado (usando `yfinance`).

2. **Procesamiento**
   - Limpieza de texto (stopwords, emojis, jerga argentina).
   - Análisis de sentimiento usando `pysentimiento` (modelo RoberTuito).
   - Agregación temporal (resumen por día).
   - Feature engineering (indicadores técnicos de mercado + volumen y sentimiento de texto).

3. **Modelado y Predicción**
   - Deep Learning: LSTM para predicción binaria (sube/baja).
   - Validación walk-forward para series temporales.
   - Comparación de performance con/sin señales de texto.

4. **Automatización**
   - Scraping histórico de Reddit cuando se necesite.
   - Procesamiento y actualización del dataset según necesidad.
   - Entrenamiento y generación de predicciones.
---

## Estructura de Carpetas

```
project-root/
│
├── data/
│ ├── preprocesada/ # Datos unificados y limpios (JSONL)
│ ├── procesada/ # Datos con análisis de sentimiento (JSONL)
│ └── historical/ # Datos históricos scrapeados (JSON/CSV por mes)
│
├── src/
│ ├── scraping/
│ │ ├── scraper_historico.py # Scraper histórico de Reddit
│ │ └── reddit_config.py # Configuración de Reddit API
│ ├── processing/
│ │ ├── etl_pipeline.py # Pipeline ETL principal
│ │ ├── sentiment_pipeline.py # Pipeline de análisis de sentimiento
│ │ ├── run_etl.py # Script CLI para ETL
│ │ └── run_sentiment.py # Script CLI para sentiment analysis
│ └── modelo/
│   ├── model_utils.py # Utilidades del modelo (MaskedSeqDataset, LSTMBinary)
│   ├── training.py # Entrenamiento con Cross-Validation
│   ├── train_final.py # Entrenamiento final con todos los datos
│   ├── optuna_search.py # Optimización de hiperparámetros
│   ├── predict.py # Generar predicciones
│   ├── plot_roc.py # Visualizar curva ROC
│   └── data/ # Datos del modelo (CSV, modelos entrenados, etc.)
│
├── notebooks/ # Experimentos y prototipos
├── logs/ # Logs de ejecución
└── README.md
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

Para obtener datos históricos de Reddit y aumentar el tamaño del dataset, puedes usar el scraper histórico:

```bash
# Scraping histórico con filtro de keywords (solo posts financieros)
python src/scraping/scraper_historico.py \
    --start-date 2020-01-01 \
    --end-date 2024-12-31

# Scraping histórico SIN filtro de keywords (todos los posts de los subreddits)
python src/scraping/scraper_historico.py \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --no-keywords-filter

# Ejecutar en background (recomendado para períodos largos)
nohup python src/scraping/scraper_historico.py \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --no-keywords-filter \
    > logs/historical_scraper_output.log 2>&1 &
```

**Opciones disponibles**:
- `--start-date YYYY-MM-DD`: Fecha de inicio (default: 2020-01-01)
- `--end-date YYYY-MM-DD`: Fecha de fin (default: hoy)
- `--no-keywords-filter`: Desactiva el filtro de keywords (obtiene todos los posts)
- `--max-posts N`: Máximo posts por subreddit por batch (default: 10000)
- `--no-comments`: No scrapear comentarios (más rápido)
- `--no-resume`: No continuar desde checkpoint (empezar desde cero)
- `--output-dir DIR`: Directorio de salida (default: data/historical)

**Características**:
- Guardado progresivo: cada batch mensual se guarda automáticamente
- Checkpoint/Resume: puede continuar desde donde quedó si se interrumpe
- Múltiples estrategias: usa `top`, `hot`, `controversial` con diferentes time_filters
- Rate limiting: respeta los límites de la API de Reddit
- Logs detallados: ver progreso en `logs/historical_scraper_YYYYMMDD.log`

**Monitorear progreso**:
```bash
# Ver logs en tiempo real
tail -f logs/historical_scraper_$(date +%Y%m%d).log

# Ver solo información importante
tail -f logs/historical_scraper_$(date +%Y%m%d).log | grep -E "(Procesando batch|posts válidos|Batch.*completado)"
```

**Datos generados**:
- `data/historical/reddit_historical_YYYY-MM.json`: JSON con todos los datos del mes
- `data/historical/reddit_historical_YYYY-MM_SUBREDDIT_posts.csv`: Posts por subreddit
- `data/historical/reddit_historical_YYYY-MM_SUBREDDIT_comments.csv`: Comentarios por subreddit
- `data/historical/scraping_checkpoint.json`: Checkpoint de progreso

### 3. Pipeline ETL

```bash
# Procesar solo datos de Reddit
python src/processing/run_etl.py \
    --reddit-only \
    --output-dir data/preprocesada \
    --format jsonl \
    --verbose

# Procesar Reddit y noticias
python src/processing/run_etl.py \
    --output-dir data/preprocesada \
    --format jsonl \
    --verbose

# Ver estadísticas
python src/processing/run_etl.py --reddit-only --stats
```
**Salida**: `data/preprocesada/unified_data_YYYYMMDD_HHMMSS.jsonl`

**Procesar datos históricos**:
```bash
# Procesar datos históricos scrapeados
python src/processing/run_etl.py \
    --reddit-only \
    --reddit-dir data/historical \
    --output-dir data/preprocesada \
    --format jsonl \
    --verbose
```

### 4. Análisis de Sentimiento

Aplica análisis de sentimiento usando `pysentimiento` (modelo RoberTuito) a los datos preprocesados

```bash
# Encontrar el archivo más reciente generado
LATEST_ETL=$(ls -t data/preprocesada/unified_data_*.jsonl | head -1)
echo "Procesando: $LATEST_ETL"

# Ejecutar análisis de sentimiento
python src/processing/run_sentiment.py \
    "$LATEST_ETL" \
    --output data/procesada/$(basename "$LATEST_ETL" .jsonl)_with_sentiment.jsonl \
    --batch-size 32 \
    --verbose
```

**Salida**: `data/procesada/unified_data_YYYYMMDD_HHMMSS_with_sentiment.jsonl`

### 5. Preparar Dataset para el Modelo

Los datasets de entrenamiento y test se generan usando el notebook `notebooks/generar_datasets.ipynb`. Este notebook:

1. Carga datos de Reddit con análisis de sentimiento
2. Extrae la probabilidad positiva de sentimiento (`pos_prob_mean`) agrupada por día
3. Descarga datos de MERVAL desde yfinance y calcula retorno logarítmico
4. Carga datos del dólar desde CSV y calcula retorno logarítmico
5. Crea la columna `prediccion` (target): 1 si MERVAL sube al día siguiente, 0 si baja
6. Guarda los datasets en formato CSV con separador `;`

**Archivos generados**:
- `data/final/sentiment_train_2023_2024.csv`: Dataset de entrenamiento (2023-2024)
- `data/final/sentiment_test_2025.csv`: Dataset de test (2025)

**Columnas del dataset**:
- `pos_prob_mean`: Media de probabilidad positiva de sentimiento por día
- `retorno_log_merval`: Retorno logarítmico del MERVAL
- `retorno_log_dolar`: Retorno logarítmico del dólar
- `prediccion`: Target (1 si MERVAL sube al día siguiente, 0 si baja, NaN para días no hábiles)

### 6. Entrenar el Modelo LSTM

El modelo LSTM se encuentra en `src/modelo/`. Para entrenarlo:

#### 6.1. Preparar datos para el modelo

Primero, copia los datasets generados al directorio del modelo:

```bash
# El notebook generar_datasets.ipynb copia automáticamente los archivos
# O manualmente:
cp data/final/sentiment_train_2023_2024.csv src/modelo/data/data_train.csv
cp data/final/sentiment_test_2025.csv src/modelo/data/data_test.csv
```

#### 6.2. Entrenamiento con Cross-Validation

```bash
cd src/modelo
python training.py
```

Este script realiza validación cruzada temporal y genera:
- `src/modelo/data/val_predictions.csv`: Predicciones de validación
- `src/modelo/data/roc_curve.png`: Curva ROC

#### 6.3. Entrenamiento Final

```bash
cd src/modelo
python train_final.py
```

Entrena el modelo con todos los datos disponibles y guarda:
- `src/modelo/data/final_model_state.pth`: Modelo entrenado
- `src/modelo/data/loss_curve_final.png`: Curva de pérdida

#### 6.4. Optimización de Hiperparámetros

```bash
cd src/modelo
python optuna_search.py [n_trials]
```

Optimiza hiperparámetros usando Optuna y genera:
- `src/modelo/data/optuna_study.db`: Base de datos con todos los trials
- `src/modelo/data/optuna_trials.csv`: Historial de trials
- `src/modelo/data/best_params.csv`: Mejores hiperparámetros
- `src/modelo/data/param_importance.png`: Importancia de parámetros
- `src/modelo/data/optimization_history.png`: Historial de optimización

**Ejecutar en background**:
```bash
cd src/modelo
nohup python optuna_search.py 200 > ../../logs/optuna.log 2>&1 &
tail -f ../../logs/optuna.log
```

#### 6.5. Generar Predicciones

```bash
cd src/modelo
python predict.py
```

Genera predicciones en el conjunto de test y guarda:
- `src/modelo/data/predictions.csv`: Predicciones con probabilidades y clases

#### 6.6. Visualizar Resultados

```bash
cd src/modelo
python plot_roc.py --input src/modelo/data/val_predictions.csv
```

---

## Script de Flujo Completo

`script_flujo_completo.sh` ejecuta ETL, análisis de sentimiento y preparación del dataset para entrenamiento.



