# MERVAL Sentiment Analysis & Prediction

<p align="center">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Optuna" src="https://img.shields.io/badge/Optuna-HP%20tuning-3A99D8">
  <img alt="pysentimiento" src="https://img.shields.io/badge/pysentimiento-RoBERTuito-orange">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
</p>

> Predicting the daily direction of the **Argentine MERVAL** index using **LSTM** networks fed with **sentiment signals** scraped from Reddit (`r/merval`, `r/argentina`, etc.).

🇬🇧 English below · 🇪🇸 Versión en español primero.

---

## 🇪🇸 Resumen

Este proyecto combina **NLP** y **series temporales** para responder una pregunta concreta: **¿el sentimiento agregado en foros financieros argentinos contiene información predictiva sobre la dirección del MERVAL al día siguiente?**

- **Tarea:** clasificación binaria (sube / baja al día siguiente).
- **Features:** probabilidad positiva de sentimiento agregada por día, retorno log de MERVAL, retorno log del dólar.
- **Modelo:** LSTM unidireccional (single layer) con dropout, entrenado con secuencias de 90 días.
- **Validación:** rolling-window cross-validation temporal (sin leakage).
- **Tuning:** Bayesian optimization con Optuna (200 trials).

### Métrica final

| Métrica | Valor |
|---------|-------|
| AUC (test, out-of-sample) | en revisión — ver `src/modelo/data/predictions.csv` |
| Matriz de confusión (test) | ver README detallado abajo |

### Pipeline

```
Reddit scraping ─► ETL + cleaning ─► sentiment (RoBERTuito) ─► daily aggregation
                                                                        │
yfinance (MERVAL) ──────────────────────────────────────────────────────┤
Dólar (CSV) ────────────────────────────────────────────────────────────┤
                                                                        ▼
                                                              Dataset (windowed)
                                                                        │
                                                                Optuna search (AUC)
                                                                        │
                                                            Rolling-CV training
                                                                        │
                                                                  Final model
                                                                        │
                                                                Test predictions
```

### Stack
PyTorch · pysentimiento (RoBERTuito) · Optuna · Pandas · yfinance · scikit-learn

### Autores
- **Sebastián Mesch Henriques** — [@SMESCH1](https://github.com/SMESCH1)
- **Leandro Carcagno** — [@lcgno](https://github.com/lcgno)

### Cómo correr
Ver sección "Flujo de trabajo" más abajo (idéntica versión en EN). Resumen:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 1) Credenciales de Reddit en .env (ver .env.example)
# 2) python src/scraping/scraper_historico.py --start-date 2020-01-01 --end-date 2024-12-31
# 3) python src/processing/run_etl.py --reddit-only --output-dir data/preprocesada --format jsonl
# 4) python src/processing/run_sentiment.py data/preprocesada/unified_data_2025.jsonl --output data/procesada/with_sentiment.jsonl
# 5) Jupyter: notebooks/generar_datasets.ipynb
# 6) cd src/modelo && python optuna_search.py 200
# 7) python training.py && python train_final.py && python predict.py
```

### Datos
Los datos crudos (>200 MB de JSON de Reddit) **no se versionan en git**. Para reproducir, correr el scraper o descargar el dataset desde el release vinculado (TODO: subir release con dataset preprocesado).

### Presentación
📊 [Presentación final del proyecto (Google Slides)](https://docs.google.com/presentation/d/e/2PACX-1vQxqObrHd2g0kOkwWzr-udYWEFjN8OG7mZkR9wlRs6HeNHhtzN6N1P0gilSl_TVJTZczMcXIHu1_fmx/pub?start=false&loop=false&delayms=3000)

---

## 🇬🇧 English

This project combines **NLP** and **time series** to answer a concrete question: **does the aggregated sentiment in Argentine financial forums contain predictive information on the next-day MERVAL direction?**

- **Task:** binary classification (up / down next day).
- **Features:** daily-aggregated positive sentiment probability, log return of MERVAL, log return of USD.
- **Model:** single-layer unidirectional LSTM with dropout, trained on 90-day sequences.
- **Validation:** rolling-window temporal cross-validation (no leakage).
- **Tuning:** Bayesian optimization with Optuna (200 trials).

### Stack
PyTorch · pysentimiento (RoBERTuito) · Optuna · Pandas · yfinance · scikit-learn

### Authors
- **Sebastián Mesch Henriques** — [@SMESCH1](https://github.com/SMESCH1)
- **Leandro Carcagno** — [@lcgno](https://github.com/lcgno)

### Repository layout

```
.
├── data/                         # local data (gitignored; see notes)
├── docs/                         # bibliography & docs
├── notebooks/
│   ├── generar_datasets.ipynb    # train/test dataset generation
│   └── analisis_lstm_completo.ipynb
├── src/
│   ├── scraping/                 # Reddit scrapers
│   ├── processing/               # ETL + sentiment pipelines
│   └── modelo/                   # LSTM, training, Optuna search, prediction
├── requirements.txt
└── README.md
```

### How to run
See the README in spanish above for the full pipeline. Quick version:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in Reddit creds
python src/scraping/scraper_historico.py --start-date 2020-01-01 --end-date 2024-12-31
python src/processing/run_etl.py --reddit-only --output-dir data/preprocesada --format jsonl
python src/processing/run_sentiment.py data/preprocesada/unified_data.jsonl --output data/procesada/with_sentiment.jsonl
# then run the notebooks / scripts in src/modelo/
```

### Results

Optimal hyperparameters found by Optuna:
- `SEQ_LEN = 90`, `HIDDEN_SIZE = 28`, `DROPOUT = 0.46`, `LEARNING_RATE = 5.7e-4`, `BIDIRECTIONAL = False`

ROC curve and confusion matrix are saved to `src/modelo/data/`.

### License
MIT — see `LICENSE`.

---

> 📜 The original detailed README (in Spanish) is preserved as `docs/README_full_ES.md` for reference.
