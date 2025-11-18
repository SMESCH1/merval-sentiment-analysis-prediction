# Especificación de Requerimientos de Software
## Proyecto: Predicción de Precios de Activos Financieros en Argentina usando Deep Learning y Datos Sociales

---

## 1. Introducción

### 1.1 Propósito
El propósito de este sistema es desarrollar una aplicación de predicción de precios de activos financieros en Argentina, integrando datos de mercado, noticias económicas y discusiones en redes sociales (Reddit).  
El sistema permitirá analizar si las señales sociales y de sentimiento aportan valor predictivo a modelos tradicionales de series temporales.

### 1.2 Alcance
El sistema:
- Obtendrá información de Reddit y de medios económicos argentinos mediante scraping.  
- Extraerá datos financieros desde fuentes abiertas (yfinance, BCRA, INDEC).  
- Procesará y limpiará textos para análisis de sentimiento y generación de embeddings.  
- Entrenará modelos predictivos (ARIMA, LSTM, Transformers).  
- Generará predicciones periódicas sobre precios de activos seleccionados.  
- Ofrecerá resultados en formato CSV, gráficos y un dashboard local.  

### 1.3 Definiciones y Acrónimos
- **Scraping**: Extracción automática de datos desde sitios web.  
- **Embeddings**: Representación vectorial de texto.  
- **LLM**: Large Language Model.  
- **LSTM**: Long Short-Term Memory, red neuronal recurrente para series temporales.  
- **TFT**: Temporal Fusion Transformer.  

---

## 2. Descripción General

### 2.1 Perspectiva del Sistema
El sistema es una aplicación modular que corre en entornos locales, sin depender de servicios en la nube pagos.  
Se compone de:
- **Scrapers diarios**: Reddit y noticias.  
- **Procesamiento de datos**: Limpieza, análisis de sentimiento y features.  
- **Modelado**: Entrenamiento y evaluación de modelos predictivos.  
- **Predicciones**: Generación de resultados periódicos.  
- **Visualización**: Dashboard interactivo en Streamlit.  

### 2.2 Funciones del Sistema
- Extracción automática de datos.  
- Procesamiento y transformación de datos en features.  
- Entrenamiento periódico de modelos predictivos.  
- Exportación de resultados en diferentes formatos.  
- Visualización local de resultados.  

### 2.3 Usuarios y Características
- **Investigadores / Data Scientists**: Configuran, entrenan y evalúan modelos.  
- **Analistas financieros**: Interpretan resultados y gráficos.  
- **Estudiantes / Académicos**: Usan el sistema como caso de estudio.  

### 2.4 Restricciones
- El sistema se ejecutará en máquinas locales (Linux preferido).  
- Se usará únicamente software open source.  
- Requiere conexión a internet para scraping y descarga de datos financieros.  

### 2.5 Suposiciones y Dependencias
- Los sitios de origen (Reddit, medios de noticias) mantienen accesibilidad pública.  
- La calidad predictiva dependerá de la disponibilidad y volumen de datos.  

---

## 3. Requerimientos Específicos

### 3.1 Requerimientos Funcionales

**RF1** – El sistema debe realizar scraping diario de Reddit.  
**RF2** – El sistema debe realizar scraping diario de noticias económicas argentinas.  
**RF3** – El sistema debe obtener datos de mercado desde APIs gratuitas (yfinance, BCRA, INDEC).  
**RF4** – El sistema debe procesar y limpiar los datos de texto.  
**RF5** – El sistema debe calcular embeddings y análisis de sentimiento de los textos.  
**RF6** – El sistema debe generar features diarios agregados (volumen, sentimiento, indicadores técnicos).  
**RF7** – El sistema debe entrenar modelos predictivos con actualización semanal o mensual.  
**RF8** – El sistema debe producir predicciones en formato CSV y gráficos.  
**RF9** – El sistema debe ofrecer un dashboard local con resultados (Streamlit).  
**RF10** – El sistema debe permitir ejecutar todo el pipeline con un solo comando (`python run_pipeline.py` o `make update`).  

---

### 3.2 Requerimientos No Funcionales

**RNF1 – Performance:** El pipeline completo no debe tardar más de 1 hora en ejecutarse en hardware estándar (8GB RAM, CPU 4 cores).  
**RNF2 – Usabilidad:** El sistema debe ser utilizable con un único comando de ejecución.  
**RNF3 – Portabilidad:** El sistema debe correr en Linux y ser portable a Windows/MacOS.  
**RNF4 – Escalabilidad:** El diseño modular debe permitir la incorporación de nuevas fuentes de datos.  
**RNF5 – Seguridad:** Los datos recolectados se guardarán en archivos locales, sin compartir con terceros.  
**RNF6 – Confiabilidad:** Los logs deben registrar errores de scraping y procesamiento.  

---

## 4. Requerimientos de Interfaz

### 4.1 Interfaces de Usuario
- **CLI (Command Line Interface)**: Ejecución de scripts y pipeline.  
- **Dashboard (Streamlit)**: Visualización interactiva de resultados.  

### 4.2 Interfaces de Hardware
- Ejecución en máquina local con al menos:
  - CPU de 4 núcleos.  
  - 8 GB de RAM.  
  - 10 GB de espacio en disco.  

### 4.3 Interfaces de Software
- Python 3.9+.  
- Librerías: `praw`, `psaw`, `newspaper3k`, `BeautifulSoup`, `transformers`, `pytorch`, `statsmodels`, `yfinance`, `streamlit`.  

---

## 5. Modelo del Sistema (Vista Simplificada)

```mermaid
flowchart LR
    Reddit[Reddit Posts & Comments] --> Scraping
    News[Noticias Económicas] --> Scraping
    Market[Datos Financieros] --> Processing
    Scraping --> Processing[Procesamiento de Datos]
    Processing --> Features[Features Diarios]
    Features --> Modeling[Modelado Predictivo]
    Modeling --> Predictions[Predicciones]
    Predictions --> CSV[Resultados CSV]
    Predictions --> Dashboard[Visualización en Streamlit]
