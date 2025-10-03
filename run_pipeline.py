"""
Placeholder del Pipeline completo del proyecto
util para correr diariamente desde local scraper y requests + 
Orquesta el pipeline completo: scraping -> procesamiento -> predicción.
"""

import subprocess
import os
import datetime

# Paths relativos
SCRAPERS = [
    "src/scraping/scraping_reddit.py",
    "src/scraping/scraping_news.py"
]
PROCESSOR = "src/processing/process_data.py"
PREDICTOR = "src/models/predict.py"

LOG_DIR = "logs"


def run_step(script_path, log_file):
    """Ejecuta un script y guarda salida en logs."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(LOG_DIR, log_file), "a", encoding="utf-8") as f:
        f.write(f"\n--- {timestamp} ---\n")
        process = subprocess.run(
            ["python", script_path],
            stdout=f,
            stderr=subprocess.STDOUT
        )
    if process.returncode == 0:
        print(f"[OK] {script_path} completado. Log: {log_file}")
    else:
        print(f"[ERROR] {script_path} falló. Revisar {log_file}")


def main():
    print("Iniciando pipeline de predicción financiera...\n")

    # 1. Scraping
    for scraper in SCRAPERS:
        run_step(scraper, "scraping.log")

    # 2. Procesamiento
    run_step(PROCESSOR, "processing.log")

    # 3. Predicción
    run_step(PREDICTOR, "predict.log")

    print("\n✅ Pipeline finalizado. Revisar carpeta 'logs' para más detalles.")


if __name__ == "__main__":
    main()
