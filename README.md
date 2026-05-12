# Detección de Fraude Financiero con Machine Learning

## Descripción

Este proyecto es una pipeline de análisis y detección de fraude financiero sobre un conjunto de transacciones. Las etapas son secuenciales y dependen unas de otras: generar EDA, preprocesar datos, entrenar el mejor modelo y evaluar el modelo final.

## Estructura del proyecto

- `src/` : scripts de la pipeline en orden numérico
- `data/` : dataset de entrada (`transacciones_fraude_bigdata.csv`)
- `outputs/graficos/` : gráficos generados por el EDA
- `outputs/modelos/` : modelo entrenado persistido
- `outputs/metricas/` : métricas de validación y evaluación
- `reports/` : informes de resultados

## Comandos

Ejecutar desde la raíz del repositorio en este orden:

```bash
python src/01_carga_eda.py
python src/02_preprocesamiento.py
python src/03_entrenamiento.py
python src/04_evaluacion.py
```

## Métodos y métricas

- Pipeline con modelos: Logistic Regression, Random Forest, Gradient Boosting y MLP
- Selección del mejor modelo según AUC-PR
- Enfoque en detección de fraude con prioridad en Recall y AUC-PR
- Evaluación final sobre el conjunto de prueba usando el modelo guardado

## Dependencias

Las dependencias están en `requirements.txt` e incluyen:

- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

## Notas importantes

- El archivo de datos se lee desde `data/transacciones_fraude_bigdata.csv`
- No ejecutar los scripts fuera de orden, ya que `03_entrenamiento.py` crea el artefacto que `04_evaluacion.py` necesita
- `transaction_id` se trata como columna no predictiva y se elimina durante el preprocesamiento
- La métrica principal es AUC-PR, no precisión absoluta
