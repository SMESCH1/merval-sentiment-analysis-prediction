# Proyecto: Predicción de Precios de Activos Financieros en Argentina usando Deep Learning y Datos Sociales

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
   - Scraping de noticias de economía y finanzas de Argentina.
   - Descarga de datos de mercado (usando `yfinance` y fuentes oficiales como BCRA o INDEC).

2. **Procesamiento**
   - Limpieza de texto (stopwords, emojis, jerga argentina).
   - Análisis de sentimiento (modelos open-source como `BETO`).
   - Embeddings para representación semántica.
   - Agregación temporal (resumen por día).
   - Feature engineering (indicadores técnicos de mercado + volumen y sentimiento de texto).

3. **Modelado y Predicción**
   - Baselines: ARIMA, regresión lineal.
   - Deep Learning: LSTM, GRU, Transformers para series temporales.
   - Comparación de performance con/sin señales de texto.

4. **Automatización**
   - Scraping diario con `cron`.
   - Procesamiento y actualización del dataset semanal/mensual.
   - Entrenamiento y generación de predicciones periódicas.
   - Dashboard local (opcional) con **Streamlit**.

---

## Estructura de Carpetas

```
project-root/
│
├── data/
│   ├── raw/          # Datos crudos (JSON/CSV de scrapers)
│   ├── processed/    # Datos limpios y features
│   └── market/       # Datos financieros descargados
│
├── src/
│   ├── scraping/
│   │   ├── scraping_reddit.py
│   │   └── scraping_news.py
│   ├── processing/
│   │   ├── clean_text.py
│   │   └── process_data.py
│   ├── models/
│   │   ├── train_baseline.py
│   │   ├── train_deep.py
│   │   └── predict.py
│   └── utils/        # Funciones auxiliares
│
├── notebooks/        # Experimentos y prototipos
├── logs/             # Logs de ejecución diaria
├── dashboard.py      # Dashboard en Streamlit
├── run_pipeline.py   # Orquestación del pipeline completo
└── README.md         # Documentación del proyecto
```

---

## Flujo de Trabajo

### 1. Ejecución de scrapers (diario)
- **Reddit** (`scraping_reddit.py`).
- **Noticias** (`scraping_news.py`).
- Programado con **cron**:

```bash

ejemplo
0 9 * * * /usr/bin/python3 /home/usuario/proyecto/src/scraping/scraping_reddit.py >> logs/reddit.log 2>&1
0 9 * * * /usr/bin/python3 /home/usuario/proyecto/src/scraping/scraping_news.py >> logs/news.log 2>&1
```

---

### 2. Procesamiento de datos (semanal/mensual)
- Limpieza de textos y normalización.
- Generación de embeddings y sentimientos.
- Agregación por día.
- Exportación a `data/processed/features.csv`.

```bash
python src/processing/process_data.py
```

---

### 3. Entrenamiento del modelo
- **Semanal o mensual** para ahorrar recursos.
- Retrain con ventana móvil (ej. últimos 6 meses).
- Guardar modelos en `models/checkpoints/`.

```bash
python src/models/train_deep.py
```

---

### 4. Generación de predicciones
- Corre `predict.py` con el último modelo entrenado.
- Output: CSV con predicciones + gráficos.

```bash
python src/models/predict.py
```

---

### 5. Visualización (opcional)
- Dashboard interactivo en **Streamlit**:

```bash
streamlit run dashboard.py
```

---

## Herramientas Open Source

- **Scraping**: `praw`, `psaw`, `newspaper3k`, `BeautifulSoup`.
- **Procesamiento de texto**: `nltk`, `spaCy`, `transformers`, `sentence-transformers`.
- **Series temporales**: `statsmodels`, `pytorch`, `pytorch-forecasting`.
- **Automatización**: `cron`, `make`, `invoke`.
- **Visualización**: `matplotlib`, `seaborn`, `streamlit`.

---

## 🚀 Configuración Inicial

### Prerrequisitos
- Python 3.11+
- Docker Desktop (para automatización)
- Cuenta de Reddit con API credentials

### 1. Clonar y Configurar el Proyecto

```bash
# Clonar el repositorio
git clone <tu-repositorio>
cd stock_price_prediction_MIA

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Credenciales de Reddit

```bash
# Ejecutar script de configuración
python setup_reddit.py
```

Este script te guiará para:
- Crear una aplicación en Reddit (https://www.reddit.com/prefs/apps)
- Configurar las credenciales en el archivo `.env`

### 3. Probar la Conexión

```bash
# Verificar que las credenciales funcionan
python test_reddit_connection.py
```

### 4. Configurar Automatización con Docker

```bash
# Instalar Docker Desktop desde:
# https://www.docker.com/products/docker-desktop

# Configurar el contenedor automatizado
python docker_setup.py
```

El contenedor Docker:
- ✅ Se ejecuta automáticamente todos los días a las 9:00 AM
- ✅ Se reinicia solo si hay problemas
- ✅ Persiste los datos en tu máquina local
- ✅ Funciona en Windows, Linux y macOS

### 5. Ejecutar Scraping Manual (Opcional)

```bash
# Ejecutar una vez para probar
python src/scraping/scraping_reddit.py

# O ejecutar el scraper diario
python src/scraping/daily_reddit_scraper.py
```

### 6. Verificar Datos

Los datos se guardan en:
- `data/raw/` - Datos crudos en JSON y CSV
- `logs/` - Logs de ejecución

---

## 📊 Uso del Sistema

### Comandos Principales

```bash
# Gestionar contenedor Docker
python docker_setup.py

# Ver logs del scraper
docker-compose logs -f

# Ejecutar scraper manualmente
docker-compose exec reddit-scraper python src/scraping/daily_reddit_scraper.py

# Detener automatización
docker-compose down
```

### Estructura de Datos

Los datos se organizan por fecha y subreddit:
```
data/raw/
├── reddit_data_20241201_090000.json          # Datos completos
├── reddit_data_20241201_090000_merval_posts.csv
├── reddit_data_20241201_090000_merval_comments.csv
├── reddit_data_20241201_090000_argentina_posts.csv
└── ...
```

---

## 🔧 Solución de Problemas

### Error de Credenciales
```bash
# Verificar archivo .env
cat .env

# Reconfigurar credenciales
python setup_reddit.py
```

### Error de Docker
```bash
# Verificar que Docker esté corriendo
docker --version
docker info

# Reiniciar contenedor
docker-compose down
docker-compose up -d
```

### Ver Logs
```bash
# Logs del contenedor
docker-compose logs -f

# Logs del scraper
tail -f logs/reddit_scraper.log
```

---

## Plan de Acción

1. **Infraestructura mínima** ✅
   - Crear carpetas `data/raw`, `data/processed`, `logs/`.
   - Implementar scrapers iniciales.

2. **Prototipo**
   - Obtener 1 semana de datos.
   - Generar features básicos.
   - Baseline: regresión lineal con sentimiento agregado.

3. **Automatización** ✅
   - Configurar Docker para scraping diario.
   - Procesamiento y actualización semanal.

4. **Iteración**
   - Añadir indicadores técnicos.
   - Implementar LSTM/Transformers.
   - Evaluar y comparar resultados.

---
