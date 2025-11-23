#!/bin/bash

# Script para ejecutar entrenamiento en background
# Uso: ./run_training.sh [script] [log_file]
# script puede ser: training.py, train_final.py, optuna_search.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parámetros
SCRIPT=${1:-"train_final.py"}  # Script a ejecutar (default: train_final.py)
LOG_FILE=${2:-"logs/training.log"}  # Archivo de log

# Crear directorio de logs si no existe
mkdir -p logs
mkdir -p data

# Activar entorno virtual si existe
if [ -d "../../venv" ]; then
    source ../../venv/bin/activate
fi

# Verificar que existe el script
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: No se encontró $SCRIPT"
    exit 1
fi

# Verificar que existe el archivo de datos
if [ ! -f "data/data_train.csv" ]; then
    echo "ERROR: No se encontró data/data_train.csv"
    echo "Por favor, ejecuta primero el notebook generar_datasets.ipynb"
    exit 1
fi

echo "=========================================="
echo "INICIANDO ENTRENAMIENTO"
echo "=========================================="
echo "Script: $SCRIPT"
echo "Log file: $LOG_FILE"
echo "Fecha inicio: $(date)"
echo "=========================================="

# Ejecutar en background con nohup
nohup python "$SCRIPT" > "$LOG_FILE" 2>&1 &

# Guardar PID
TRAINING_PID=$!
echo "Proceso iniciado con PID: $TRAINING_PID"
echo "PID guardado en: logs/training.pid"
echo "$TRAINING_PID" > logs/training.pid

echo ""
echo "Para ver el progreso en tiempo real:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Para detener el proceso:"
echo "  kill $TRAINING_PID"
echo ""
echo "Para verificar si está corriendo:"
echo "  ps aux | grep $SCRIPT"

