import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    precision_recall_curve
)


# 1. CONFIGURACIÓN
UMBRAL = 0.30

os.makedirs("outputs/graficos", exist_ok=True)
os.makedirs("outputs/metricas", exist_ok=True)

# 2. CARGAR DATASET
df = pd.read_csv("data/transacciones_fraude_bigdata.csv")

X = df.drop(columns=["fraud", "transaction_id"])
y = df["fraud"]

# 3. RECREAR PARTICIÓN TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. CARGAR MEJOR MODELO
modelo = joblib.load("outputs/modelos/mejor_modelo_fraude.pkl")

print("\nMODELO CARGADO CORRECTAMENTE")

# 5. PREDICCIONES EN TEST
y_prob = modelo.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= UMBRAL).astype(int)

# 6. MÉTRICAS FINALES
auc_roc = roc_auc_score(y_test, y_prob)
auc_pr = average_precision_score(y_test, y_prob)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

metricas_finales = {
    "umbral": UMBRAL,
    "AUC_ROC": auc_roc,
    "AUC_PR": auc_pr,
    "Recall": recall,
    "Precision": precision,
    "F1": f1
}

print("\nMÉTRICAS FINALES EN TEST:")
for metrica, valor in metricas_finales.items():
    print(f"{metrica}: {valor:.4f}" if isinstance(valor, float) else f"{metrica}: {valor}")

print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"], zero_division=0))

# 7. MATRIZ DE CONFUSIÓN
cm = confusion_matrix(y_test, y_pred)

print("\nMATRIZ DE CONFUSIÓN:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraude"],
    yticklabels=["Normal", "Fraude"]
)

plt.title("Matriz de Confusión - Test")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.savefig("outputs/graficos/matriz_confusion_test.png", dpi=300)
plt.show()

# 8. CURVA ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC-ROC = {auc_roc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Curva ROC - Test")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/graficos/curva_roc_test.png", dpi=300)
plt.show()


# 9. CURVA PRECISION-RECALL
precisions, recalls, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions, label=f"AUC-PR = {auc_pr:.4f}")
plt.title("Curva Precision-Recall - Test")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/graficos/curva_pr_test.png", dpi=300)
plt.show()


# 10. GUARDAR MÉTRICAS
with open("outputs/metricas/metricas_finales_test.json", "w", encoding="utf-8") as f:
    json.dump(metricas_finales, f, indent=4, ensure_ascii=False)

pd.DataFrame([metricas_finales]).to_csv(
    "outputs/metricas/metricas_finales_test.csv",
    index=False
)

print("\nMÉTRICAS FINALES GUARDADAS EN outputs/metricas/")
print("GRÁFICOS GUARDADOS EN outputs/graficos/")
print("\nEVALUACIÓN FINALIZADA CORRECTAMENTE")