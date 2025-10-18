# Documentación de Integración con Reddit API

## 🚀 Configuración Inicial

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Configuración de Credenciales de Reddit
```bash
python setup_reddit.py
```

### 3. Prueba de Conexión
```bash
python test_reddit_connection.py
```

### 4. Ejecución del Scraper
```bash
python src/scraping/scraping_reddit.py
```

### 5. Automatización Diaria (Opcional)
```bash
python setup_cron.py
```

---

## 📊 Subreddits Monitoreados

El sistema recolecta datos de los siguientes subreddits argentinos relacionados con finanzas:

1. **r/merval** - Principal subreddit del mercado bursátil argentino
2. **r/argentina** - Subreddit general de Argentina (incluye discusiones financieras)
3. **r/argentinacrypto** - Discusiones sobre criptomonedas argentinas
4. **r/argentinaeconomia** - Discusiones sobre economía argentina
5. **r/CryptoArgentina** - Subreddit alternativo de criptomonedas
6. **r/InversionesArg** - Inversiones argentinas
7. **r/finanzaspersonales** - Finanzas personales (español)

---

## 🔍 Datos Recolectados

### Posts
- Título y contenido del texto
- Información del autor
- Puntuación (upvotes/downvotes)
- Número de comentarios
- Timestamp de creación
- Subreddit de origen
- URL y permalink

### Comentarios
- Texto del comentario
- Información del autor
- Puntuación
- Timestamp de creación
- Información del post padre
- Subreddit de origen

### Filtrado
Solo se recolectan posts y comentarios que contengan palabras clave relacionadas con finanzas:
- `merval`, `acciones`, `bonos`, `dólar`, `peso`, `inflación`
- `crypto`, `bitcoin`, `ethereum`, `criptomonedas`
- `broker`, `trading`, `inversión`, `portfolio`
- Y muchos más términos financieros argentinos...

---

## 📁 Estructura de Almacenamiento

### Estructura de Archivos
```
data/
└── raw/
    ├── reddit_data_YYYYMMDD_HHMMSS.json          # Datos completos (JSON)
    ├── reddit_data_YYYYMMDD_HHMMSS_merval_posts.csv
    ├── reddit_data_YYYYMMDD_HHMMSS_merval_comments.csv
    ├── reddit_data_YYYYMMDD_HHMMSS_argentina_posts.csv
    └── ... (archivos CSV para cada subreddit)
```

### Formato de Datos

#### Formato JSON
```json
{
  "merval": {
    "posts": [...],
    "comments": [...],
    "total_posts": 45,
    "total_comments": 123
  },
  "argentina": {
    "posts": [...],
    "comments": [...],
    "total_posts": 23,
    "total_comments": 67
  }
}
```

#### Formato CSV
Cada archivo CSV contiene datos estructurados con columnas:
- `id`, `title`, `text`, `author`, `subreddit`
- `score`, `upvote_ratio`, `created_utc`
- `url`, `permalink`, `type`

---

## 🤖 Automatización

### Scraping Diario
Configurar recolección automatizada diaria a las 9:00 AM:
```bash
python setup_cron.py
```

### Ejecución Manual
```bash
# Ejecutar una vez
python src/scraping/scraping_reddit.py

# Ejecutar automatización diaria
python src/scraping/daily_reddit_scraper.py
```

### Trabajo Cron
El script de automatización crea un trabajo cron:
```bash
0 9 * * * cd /ruta/al/proyecto && python3 src/scraping/daily_reddit_scraper.py >> logs/cron.log 2>&1
```

---

## 📈 Ejemplos de Uso

### Scraping Básico
```python
from src.scraping.scraping_reddit import RedditScraper

# Inicializar scraper
scraper = RedditScraper()

# Scrapear subreddit específico
posts = scraper.scrape_subreddit_posts('merval', limit=100)
comments = scraper.scrape_subreddit_comments('merval', limit=100)

# Scrapear todos los subreddits
all_data = scraper.scrape_all_subreddits()
```

### Procesamiento de Datos
```python
import pandas as pd

# Cargar datos recolectados
df_posts = pd.read_csv('data/raw/reddit_data_20241201_090000_merval_posts.csv')
df_comments = pd.read_csv('data/raw/reddit_data_20241201_090000_merval_comments.csv')

# Analizar sentimiento, extraer features, etc.
```

---

## 🔧 Configuración

### Límites de Velocidad
El scraper respeta los límites de velocidad de Reddit:
- 60 solicitudes por minuto
- Retraso de 1 segundo entre solicitudes
- Reintento automático en errores de límite de velocidad

### Personalización
Editar `src/scraping/reddit_config.py` para:
- Agregar/remover subreddits
- Modificar palabras clave financieras
- Ajustar límites de velocidad
- Cambiar parámetros de recolección de datos

---

## 📋 Logging

### Archivos de Log
- `logs/reddit_scraper.log` - Logs generales del scraper
- `logs/reddit_daily_YYYYMMDD.log` - Logs de automatización diaria
- `logs/cron.log` - Logs de ejecución de trabajos cron

### Niveles de Log
- **INFO**: Progreso general y estado
- **DEBUG**: Información detallada de recolección
- **ERROR**: Errores y fallos
- **WARNING**: Límites de velocidad y reintentos

---

## 🛠️ Solución de Problemas

### Problemas Comunes

#### 1. Errores de Autenticación
```
Error: Invalid credentials
```
**Solución**: Verificar las credenciales de la API de Reddit en el archivo `.env`

#### 2. Errores de Límite de Velocidad
```
Error: 429 Too Many Requests
```
**Solución**: El scraper maneja esto automáticamente con retrasos

#### 3. Errores de Acceso a Subreddit
```
Error: 403 Forbidden
```
**Solución**: Algunos subreddits pueden ser privados o restringidos

#### 4. Errores de Red
```
Error: Connection timeout
```
**Solución**: Verificar la conexión a internet y el estado de Reddit

### Modo Debug
Habilitar logging detallado:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

---

## 📊 Análisis de Datos

### Script de Análisis de Ejemplo
```python
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Cargar datos
df = pd.read_csv('data/raw/reddit_data_20241201_090000_merval_posts.csv')

# Convertir timestamp
df['created_utc'] = pd.to_datetime(df['created_utc'])

# Conteos diarios de posts
daily_posts = df.groupby(df['created_utc'].dt.date).size()
daily_posts.plot(title='Posts Diarios en r/merval')
plt.show()

# Análisis de sentimiento (requiere configuración adicional)
# from textblob import TextBlob
# df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
```

---

## 🔒 Seguridad

### Protección de Credenciales
- Nunca commitear el archivo `.env` al control de versiones
- Agregar `.env` a `.gitignore`
- Usar variables de entorno en producción

### Privacidad de Datos
- Los datos recolectados son contenido público de Reddit
- No se almacena información privada de usuarios
- Respeta los términos de servicio de Reddit

---

## 📚 Próximos Pasos

1. **Procesamiento de Texto**: Configurar análisis de sentimiento y preprocesamiento de texto
2. **Ingeniería de Features**: Extraer características significativas de los datos de texto
3. **Integración de Series Temporales**: Combinar con datos de mercado
4. **Entrenamiento de Modelos**: Usar datos recolectados para modelos de predicción
5. **Dashboard**: Crear dashboard de visualización

---

## 🆘 Soporte

Si encuentras problemas:
1. Revisar los logs en el directorio `logs/`
2. Verificar las credenciales de la API de Reddit
3. Probar la conexión con `python test_reddit_connection.py`
4. Verificar el estado de la API de Reddit en [https://redditstatus.com](https://redditstatus.com)

