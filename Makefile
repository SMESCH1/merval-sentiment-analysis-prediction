# Makefile - Orquestación del pipeline financiero
# corre todo el pipeline desde bash con 
# `make update
# sino solo el scraper: make scraper, etc
# se puede integrar cron dentro de este makefile


PYTHON=python

# Paths
SCRAPING_REDDIT=src/scraping/scraping_reddit.py
SCRAPING_NEWS=src/scraping/scraping_news.py
PROCESS=src/processing/process_data.py
PREDICT=src/models/predict.py
PIPELINE=run_pipeline.py

# Logs
LOG_DIR=logs

# Crear carpeta de logs si no existe
$(LOG_DIR):
	mkdir -p $(LOG_DIR)

# === Targets ===

scrape: $(LOG_DIR)
	@echo "🚀 Ejecutando scraping de Reddit y Noticias..."
	$(PYTHON) $(SCRAPING_REDDIT) >> $(LOG_DIR)/scraping.log 2>&1
	$(PYTHON) $(SCRAPING_NEWS) >> $(LOG_DIR)/scraping.log 2>&1
	@echo "✅ Scraping completado. Revisar logs en $(LOG_DIR)/scraping.log"

process: $(LOG_DIR)
	@echo "⚙️ Procesando datos..."
	$(PYTHON) $(PROCESS) >> $(LOG_DIR)/processing.log 2>&1
	@echo "✅ Procesamiento completado. Revisar logs en $(LOG_DIR)/processing.log"

predict: $(LOG_DIR)
	@echo "📈 Generando predicciones..."
	$(PYTHON) $(PREDICT) >> $(LOG_DIR)/predict.log 2>&1
	@echo "✅ Predicciones generadas. Revisar logs en $(LOG_DIR)/predict.log"

update: $(LOG_DIR)
	@echo "🔄 Corriendo pipeline completo..."
	$(PYTHON) $(PIPELINE)
	@echo "✅ Pipeline completo ejecutado."

clean:
	@echo "🧹 Limpiando logs..."
	rm -rf $(LOG_DIR)/*
	@echo "✅ Logs eliminados."
