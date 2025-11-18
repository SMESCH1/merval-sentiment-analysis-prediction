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
│ ├── procesamiento/
│ │ ├── etl_pipeline.py # Pipeline ETL principal
│ │ ├── sentiment_pipeline.py # Pipeline de análisis de sentimiento
│ │ ├── run_etl.py # Script CLI para ETL
│ │ └── run_sentiment.py # Script CLI para sentiment analysis
│ └── LSTM/
│ ├── training_data.py # Combina sentiment + datos financieros
│ ├── train_boolean_lstm.py # Entrenamiento del modelo LSTM
│ └── dataset_con_sentiment.csv # Dataset final para entrenar
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
python src/procesamiento/run_etl.py \
    --reddit-only \
    --output-dir data/preprocesada \
    --format jsonl \
    --verbose

# Procesar Reddit y noticias
python src/procesamiento/run_etl.py \
    --output-dir data/preprocesada \
    --format jsonl \
    --verbose

# Ver estadísticas
python src/procesamiento/run_etl.py --reddit-only --stats
```
**Salida**: `data/preprocesada/unified_data_YYYYMMDD_HHMMSS.jsonl`

**Procesar datos históricos**:
```bash
# Procesar datos históricos scrapeados
python src/procesamiento/run_etl.py \
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
python src/procesamiento/run_sentiment.py \
    "$LATEST_ETL" \
    --output data/procesada/$(basename "$LATEST_ETL" .jsonl)_with_sentiment.jsonl \
    --batch-size 32 \
    --verbose
```

**Salida**: `data/procesada/unified_data_YYYYMMDD_HHMMSS_with_sentiment.jsonl`

### 5. Preparar Dataset para LSTM

Combinar los datos con sentiment analysis con datos financieros del MERVAL para crear el dataset de entrenamiento

```bash
# Tarer el archivo de más reciente
LATEST_SENTIMENT=$(ls -t data/procesada/*_with_sentiment.jsonl | head -1)
echo "Combinando: $LATEST_SENTIMENT"

# Combinar sentiment con datos financieros
python src/LSTM/training_data.py \
    --sentiment-jsonl "$LATEST_SENTIMENT" \
    --output-csv src/LSTM/dataset_con_sentiment.csv \
    --merval-ticker "^MERV"
```

**Salida**: `src/LSTM/dataset_con_sentiment.csv` con las siguientes columnas:
- `retorno_log_merval`: Retorno logarítmico del MERVAL
- `sentiment_total_records`: Total de registros con sentiment ese día
- `sentiment_pos_count`, `sentiment_neg_count`, `sentiment_neu_count`: Conteos de sentimiento
- `sentiment_score`: Score normalizado (-1 a 1)
- `sentiment_pos_ratio`, `sentiment_neg_ratio`, `sentiment_neu_ratio`: Proporciones
- `sentiment_avg_confidence`, `sentiment_max_confidence`, `sentiment_min_confidence`: Confianza del modelo
- `reddit_avg_score`, `reddit_total_score`, `reddit_max_score`: Métricas de Reddit
- `booleano_merval`: Target (1 si sube, 0 si baja)

### 6. Entrenar el LSTM

Entrena el modelo LSTM con validación walk-forward

```bash
# Entrenamiento con parámetros por defecto (requiere al menos ~480 días de datos)
python src/LSTM/train_boolean_lstm.py \
    --jsonl-auto \
    --lookback 20 \
    --hidden-size 64 \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --device cpu
```

**Para datasets pequeños**, ajusta los parámetros:

```bash
# Para datasets pequeños (< 100 días)
python src/LSTM/train_boolean_lstm.py \
    --csv-path src/LSTM/dataset_con_sentiment.csv \
    --lookback 5 \
    --initial-train-size 15 \
    --test-window 5 \
    --epochs 10 \
    --batch-size 4 \
    --device cpu
```

**Parámetros importantes**:
- `--lookback`: Días históricos para crear secuencias (default: 20)
- `--initial-train-size`: Secuencias iniciales para entrenar (default: 400)
- `--test-window`: Secuencias para test en cada fold (default: 60)
- `--epochs`: Número de épocas de entrenamiento (default: 20)
- `--batch-size`: Tamaño del batch (default: 64)

**Requisitos mínimos de datos**:
- Mínimo total: `lookback + initial_train_size + test_window` días
- Ejemplo con defaults: `20 + 400 + 60 = 480` días
- Ejemplo con parámetros reducidos: `5 + 15 + 5 = 25` días

---

## Script de Flujo Completo

`script_flujo_completo.sh` ejecuta ETL, análisis de sentimiento y preparación del dataset para entrenamiento.



