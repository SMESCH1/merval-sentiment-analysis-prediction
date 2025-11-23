#!/bin/bash

# Script para detener la optimización de hiperparámetros

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="logs/optuna.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Deteniendo proceso con PID: $PID"
        kill "$PID"
        rm "$PID_FILE"
        echo "Proceso detenido"
    else
        echo "El proceso ya no está corriendo"
        rm "$PID_FILE"
    fi
else
    echo "No se encontró archivo PID. Buscando procesos de optuna..."
    PIDS=$(ps aux | grep optuna_search.py | grep -v grep | awk '{print $2}')
    if [ -z "$PIDS" ]; then
        echo "No hay procesos de optimización corriendo"
    else
        echo "Procesos encontrados: $PIDS"
        read -p "¿Deseas detenerlos? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            for pid in $PIDS; do
                kill "$pid"
                echo "Proceso $pid detenido"
            done
        fi
    fi
fi

