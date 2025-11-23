# Modelo LSTM para Predicción de MERVAL

## Descripción

Modelo LSTM para predecir si el MERVAL subirá o bajará al día siguiente basándose en:
- Sentimiento de Reddit (probabilidad positiva)
- Retorno logarítmico del MERVAL
- Retorno logarítmico del dólar

## Estructura de Archivos

```
src/modelo/
├── model_utils.py          # Utilidades del modelo (MaskedSeqDataset, LSTMBinary, etc.)
├── training.py             # Entrenamiento con Cross-Validation
├── train_final.py          # Entrenamiento final con todos los datos
├── optuna_search.py        # Optimización de hiperparámetros con Optuna
├── predict.py              # Generar predicciones
├── plot_roc.py            # Visualizar curva ROC
├── run_optimization.sh    # Script para ejecutar optimización en background
├── stop_optimization.sh   # Script para detener optimización
├── run_training.sh        # Script para ejecutar entrenamiento en background
└── data/                  # Directorio con datos de entrenamiento y test
    ├── data_train.csv     # Dataset de entrenamiento (2023-2024)
    ├── data_test.csv      # Dataset de test (2025)
    └── optuna_study.db    # Base de datos de Optuna (se crea automáticamente)
```

## Requisitos

1. **Datos preparados**: Los archivos CSV deben estar en `data/` con el formato correcto:
   - Separador: `;` (punto y coma)
   - Columna target: `prediccion` (0 o 1, o NaN para días no hábiles)
   - Features: `pos_prob_mean`, `retorno_log_merval`, `retorno_log_dolar`

2. **Entorno virtual activado**:
   ```bash
   source ../../venv/bin/activate
   ```

## Uso

### 1. Preparar Datos

Primero ejecuta el notebook `notebooks/generar_datasets.ipynb` para generar los datasets. La última celda copiará automáticamente los archivos a `src/modelo/data/`.

### 2. Entrenamiento con Cross-Validation

```bash
cd src/modelo
python training.py
```

O en background:
```bash
./run_training.sh training.py logs/training.log
tail -f logs/training.log  # Ver progreso
```

### 3. Entrenamiento Final

```bash
python train_final.py
```

O en background:
```bash
./run_training.sh train_final.py logs/train_final.log
tail -f logs/train_final.log
```

### 4. Optimización de Hiperparámetros

#### Opción A: Ejecutar directamente
```bash
python optuna_search.py
```

#### Opción B: Ejecutar en background (RECOMENDADO)
```bash
# Iniciar optimización (200 trials por defecto)
./run_optimization.sh

# O especificar número de trials
./run_optimization.sh 500

# Ver progreso en tiempo real
tail -f logs/optuna_optimization.log

# Detener optimización
./stop_optimization.sh
```

#### Opción C: Usar nohup directamente
```bash
# Iniciar
nohup python optuna_search.py > logs/optuna.log 2>&1 &

# Ver progreso
tail -f logs/optuna.log

# Detener (encontrar PID primero)
ps aux | grep optuna_search.py
kill <PID>
```

#### Opción D: Usar screen (permite reconectar)
```bash
# Crear sesión screen
screen -S optuna

# Dentro de screen, ejecutar
python optuna_search.py

# Desconectar: Ctrl+A, luego D
# Reconectar: screen -r optuna
# Ver todas las sesiones: screen -ls
```

### 5. Generar Predicciones

```bash
python predict.py
```

## Resultados de Optimización

Después de ejecutar `optuna_search.py`, encontrarás:

- `data/optuna_trials.csv`: Historial completo de todos los trials
- `data/best_params.csv`: Mejores hiperparámetros encontrados
- `data/param_importance.png`: Gráfico de importancia de parámetros
- `data/optimization_history.png`: Historial de optimización
- `data/optuna_study.db`: Base de datos SQLite con todos los trials

## Verificar Progreso

### Ver logs en tiempo real:
```bash
tail -f logs/optuna_optimization.log
```

### Ver procesos corriendo:
```bash
ps aux | grep optuna_search.py
ps aux | grep python
```

### Verificar base de datos de Optuna:
```python
import optuna
study = optuna.load_study(study_name='lstm_stock_optimization', 
                          storage='sqlite:///data/optuna_study.db')
print(f"Trials completados: {len(study.trials)}")
print(f"Mejor AUC: {study.best_value:.4f}")
print(f"Mejores parámetros: {study.best_params}")
```

## Notas Importantes

1. **Días no hábiles**: El modelo automáticamente filtra días donde `prediccion` es NaN (fines de semana, feriados).

2. **Features con NaN**: El modelo rellena automáticamente con 0 (ver `training.py` línea 48).

3. **GPU**: Si tienes GPU disponible, el modelo la usará automáticamente.

4. **Interrupción**: Puedes interrumpir la optimización en cualquier momento. Optuna guarda el progreso en la base de datos y puedes continuar después.

5. **Reanudar optimización**: Si ejecutas `optuna_search.py` de nuevo, continuará desde donde se quedó (gracias a `load_if_exists=True`).

## Troubleshooting

### Error: "No se encontró data/data_train.csv"
- Ejecuta primero el notebook `generar_datasets.ipynb`
- Verifica que la celda 11 del notebook copió los archivos correctamente

### Error: "CUDA out of memory"
- Reduce `BATCH_SIZE` en los scripts
- Reduce `N_TRIALS` si estás optimizando
- Usa CPU: `export CUDA_VISIBLE_DEVICES=""` antes de ejecutar

### Optimización muy lenta
- Reduce `N_TRIALS`
- Reduce `n_epochs` en `optuna_search.py`
- Usa menos folds en cross-validation

