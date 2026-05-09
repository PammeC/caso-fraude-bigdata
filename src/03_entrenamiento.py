import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    precision_score
)


# 1. CARGAR DATASET
df = pd.read_csv("data/transacciones_fraude_bigdata.csv")


# 2. DEFINIR X e y


X = df.drop(columns=["fraud", "transaction_id"])
y = df["fraud"]


# 3. TRAIN / TEST SPLIT 80/20


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# 4. COLUMNAS NUMÉRICAS Y CATEGÓRICAS


num_cols = [
    "transaction_amount_usd",
    "transaction_hour",
    "day_of_week",
    "customer_tenure_months",
    "prev_transactions_7d",
    "previous_fraud_count",
    "location_risk_score",
    "is_international",
    "distance_from_home_km"
]

cat_cols = [
    "device_type",
    "merchant_category",
    "channel",
    "country"
]


# 5. PREPROCESAMIENTO


preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])


# 6. DEFINIR MODELOS


scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nRatio de desbalance: {scale:.1f}x")

modelos = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        C=0.5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=8,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    ),

    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
}


# 7. VALIDACIÓN CRUZADA ESTRATIFICADA


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

UMBRAL = 0.30

resultados_cv = {}

print("\nINICIANDO VALIDACIÓN CRUZADA...\n")

for nombre, modelo in modelos.items():

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", modelo)
    ])

    auc_roc_list = []
    auc_pr_list = []
    recall_list = []
    f1_list = []
    precision_list = []

    for train_idx, val_idx in cv.split(X_train, y_train):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        sample_weight = np.where(y_tr == 1, scale, 1.0)

        if nombre in ["Gradient Boosting", "MLP Neural Network"]:
            pipe.fit(X_tr, y_tr, clf__sample_weight=sample_weight)
        else:
            pipe.fit(X_tr, y_tr)

        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= UMBRAL).astype(int)

        auc_roc_list.append(roc_auc_score(y_val, y_prob))
        auc_pr_list.append(average_precision_score(y_val, y_prob))
        recall_list.append(recall_score(y_val, y_pred, zero_division=0))
        f1_list.append(f1_score(y_val, y_pred, zero_division=0))
        precision_list.append(precision_score(y_val, y_pred, zero_division=0))

    resultados_cv[nombre] = {
        "AUC-ROC": np.mean(auc_roc_list),
        "AUC-PR": np.mean(auc_pr_list),
        "Recall": np.mean(recall_list),
        "F1": np.mean(f1_list),
        "Precision": np.mean(precision_list)
    }

    print(f"\n{nombre}")
    for metrica, valor in resultados_cv[nombre].items():
        print(f"{metrica}: {valor:.4f}")


# 8. TABLA DE RESULTADOS


resultados_df = pd.DataFrame(resultados_cv).T.round(4)

print("\nTABLA RESUMEN:")
print(resultados_df.sort_values(by="AUC-PR", ascending=False))


# 9. SELECCIONAR MEJOR MODELO


mejor_modelo_nombre = resultados_df.sort_values(
    by=["AUC-PR", "Recall"],
    ascending=False
).index[0]

print("\nMEJOR MODELO SELECCIONADO:")
print(mejor_modelo_nombre)


# 10. ENTRENAR MEJOR MODELO CON TODO TRAIN


mejor_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", modelos[mejor_modelo_nombre])
])

sample_weight_train = np.where(y_train == 1, scale, 1.0)

if mejor_modelo_nombre in ["Gradient Boosting", "MLP Neural Network"]:
    mejor_pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
else:
    mejor_pipeline.fit(X_train, y_train)

print("\nMEJOR MODELO ENTRENADO CON TODO TRAIN")


# 11. GUARDAR RESULTADOS Y MODELO


os.makedirs("outputs/modelos", exist_ok=True)
os.makedirs("outputs/metricas", exist_ok=True)

joblib.dump(
    mejor_pipeline,
    "outputs/modelos/mejor_modelo_fraude.pkl"
)

resultados_df.to_csv(
    "outputs/metricas/resultados_validacion_cruzada.csv"
)

with open("outputs/metricas/mejor_modelo.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "mejor_modelo": mejor_modelo_nombre,
            "umbral": UMBRAL,
            "metricas": resultados_df.loc[mejor_modelo_nombre].to_dict()
        },
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nMODELO GUARDADO EN outputs/modelos/mejor_modelo_fraude.pkl")
print("RESULTADOS GUARDADOS EN outputs/metricas/")
print("\nENTRENAMIENTO FINALIZADO CORRECTAMENTE")