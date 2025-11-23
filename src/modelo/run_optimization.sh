#!/bin/bash

# Script para ejecutar optimización de hiperparámetros en background
# Uso: ./run_optimization.sh [n_trials] [log_file]

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parámetros
N_TRIALS=${1:-200}  # Número de trials (default: 200)
LOG_FILE=${2:-"logs/optuna_optimization_$(date +%Y%m%d_%H%M%S).log"}  # Archivo de log con timestamp

# Crear directorio de logs si no existe
mkdir -p logs
mkdir -p data

# Activar entorno virtual si existe
if [ -d "../../venv" ]; then
    source ../../venv/bin/activate
fi

# Verificar que existe el archivo de datos
if [ ! -f "data/data_train.csv" ]; then
    echo "ERROR: No se encontró data/data_train.csv"
    echo "Por favor, ejecuta primero el notebook generar_datasets.ipynb"
    exit 1
fi

echo "=========================================="
echo "INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS"
echo "=========================================="
echo "Número de trials: $N_TRIALS"
echo "Log file: $LOG_FILE"
echo "Fecha inicio: $(date)"
echo "=========================================="

# Ejecutar optimización en background con nohup
nohup python optuna_search.py "$N_TRIALS" > "$LOG_FILE" 2>&1 &

# Guardar PID
OPTUNA_PID=$!
echo "Proceso iniciado con PID: $OPTUNA_PID"
echo "PID guardado en: logs/optuna.pid"
echo "$OPTUNA_PID" > logs/optuna.pid

echo ""
echo "Para ver el progreso en tiempo real:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Para detener el proceso:"
echo "  kill $OPTUNA_PID"
echo "  o usar: ./stop_optimization.sh"
echo ""
echo "Para verificar si está corriendo:"
echo "  ps aux | grep optuna_search.py"

