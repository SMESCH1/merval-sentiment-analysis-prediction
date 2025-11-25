import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def main():
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Genera curva ROC desde predicciones")
    parser.add_argument("--input", default="src/modelo/data/val_predictions.csv", 
                       help="CSV con columnas prob,label (de training.py) o pred_prob,prediccion (de predict.py)")
    parser.add_argument("--output", default="src/modelo/data/roc_curve.png", help="Ruta de salida para la imagen PNG")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    
    # Detectar formato del CSV
    if "prob" in df.columns and "label" in df.columns:
        # Formato de training.py (val_predictions.csv)
        y_true = df["label"].to_numpy()
        y_score = df["prob"].to_numpy()
        source = "validacion cruzada"
    elif "pred_prob" in df.columns and "prediccion" in df.columns:
        # Formato de predict.py (predictions.csv)
        df_clean = df.dropna(subset=["pred_prob", "prediccion"])
        y_true = df_clean["prediccion"].astype(float).to_numpy()
        y_score = df_clean["pred_prob"].to_numpy()
        source = "conjunto de test"
    else:
        raise ValueError(
            "El CSV debe contener:\n"
            "  - 'prob' y 'label' (salida de training.py), o\n"
            "  - 'pred_prob' y 'prediccion' (salida de predict.py)"
        )

    if len(y_true) == 0:
        raise ValueError("No hay datos validos para graficar")
    
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("No hay ambas clases en las etiquetas; no se puede calcular ROC")

    fpr, tpr, ths = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Mostrar tabla de thresholds
    table = pd.DataFrame({"thr": ths, "fpr": fpr, "tpr": tpr})
    print(table.head(10), "\n...\n", table.tail())

    # Graficar
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", color="C0")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Azar")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC ({source})")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Guardado ROC en {args.output} (AUC={roc_auc:.4f})")


if __name__ == "__main__":
    main()
