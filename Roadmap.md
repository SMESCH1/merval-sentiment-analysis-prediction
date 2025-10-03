# Roadmap del Proyecto: Predicción de Precios Financieros con Datos de Reddit y Noticias (Argentina)

## Fase 1. Definición y alcance


- Definir arquitectura (LSTM, Transformer, otra)
- Definir modelo baseline

- Definir activos a predecir (acciones argentinas, bonos, dólar, criptomonedas).
- Elegir horizonte temporal (ejemplo: 1 día, 1 semana).
- Seleccionar subreddits relevantes (r/merval, r/argentina, r/argentinacrypto, r/argentinaeconomia).
- Seleccionar medios/noticias de economía y finanzas (ej: Ámbito, Cronista, Infobae, La Nación economía).
- Establecer métricas de éxito:
  - RMSE / MAE para predicción de precios.
  - Accuracy / F1 para clasificación de sentimiento.

## Fase 2. Recolección de datos
1. Reddit
   - Usar API oficial (`praw` o `psaw`) para recolectar posts y comentarios.
   - Guardar: texto, fecha, autor, número de votos, subreddit.
2. Noticias financieras argentinas
   - Scraping de medios locales (ej: Ámbito, Cronista).
   - Guardar: titular, contenido, fecha, fuente.
   - Opcional: usar RSS feeds si están disponibles.
3. Datos financieros históricos
   - Yahoo Finance (`yfinance`) para precios históricos (OHLC, volumen).
   - APIs de brokers locales (ByMA, Rava).
   - Variables macroeconómicas (INDEC, BCRA).

## Fase 3. Procesamiento de texto
- Limpieza de texto: quitar stopwords, emojis, links.
- Normalización del español rioplatense (jerga, modismos).
- Extracción de features:
  - Embeddings preentrenados en español (BETO, SBERT-multilingual).
  - Clasificación de sentimiento (fine-tune ligero con LoRA).
  - Opcional: topic modeling para categorizar noticias y posts.

## Fase 4. Integración de datos
- Alinear información de texto con series temporales (por día o semana).
- Crear dataset final con:
  - Datos financieros (precios, volumen, indicadores técnicos).
  - Señales de Reddit (sentimiento agregado, volumen de menciones).
  - Señales de noticias (sentimiento, frecuencia de menciones, fuentes).
  - Variables macroeconómicas adicionales.

## Fase 5. Modelado
1. Baseline
   - ARIMA, regresión lineal con features básicos.
2. Deep Learning
   - LSTM / GRU para series temporales.
   - Temporal Fusion Transformer (TFT).
   - Modelos híbridos que incluyan embeddings de Reddit + embeddings de noticias como inputs adicionales.

## Fase 6. Evaluación
- Backtesting con ventanas temporales.
- Comparación de modelos con y sin datos de texto (solo precios vs precios + Reddit + noticias).
- Visualización de resultados:
  - Gráficos de predicciones vs reales.
  - Importancia de features.

## Fase 7. Deployment (opcional)
- API con FastAPI/Flask que tome datos de Reddit + noticias + mercado y devuelva predicciones.
- Dashboard en Streamlit o Gradio para visualización interactiva.

---





Extracción del Feature de Sentimiento (Unidad 6 - LLMs/RAGs):

Data: Utiliza la información de sentimiento (ej. volumen de menciones, polaridad) extraída de Twitter/Reddit sobre términos clave argentinos (dólar, Merval, BCRA).

Modelo: Puedes usar un LLM pre-entrenado en español (como un BERT o un Transformer) y ajustarlo (fine-tuning) para la clasificación de sentimiento financiero argentino (alcista, bajista, neutral).

Modelado de la Serie de Tiempo (Unidad 5 - LSTMs/Transformers):

Data: Combina series de tiempo históricas del precio del activo (Merval, dólar) con indicadores técnicos (RSI, Media Móvil) y, crucialmente, el feature de sentimiento generado en el paso 1.


Modelo: Entrena un LSTM o una arquitectura de 

Transformers  adaptada para series de tiempo para predecir el precio futuro del activo, considerando el sentimiento como una entrada clave.

