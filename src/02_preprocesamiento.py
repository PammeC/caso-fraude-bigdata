import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. CARGAR DATASET
df = pd.read_csv("data/transacciones_fraude_bigdata.csv")

print("\nDATASET CARGADO CORRECTAMENTE")

# 2. DEFINIR X e y
# Variables predictoras
X = df.drop(columns=['fraud', 'transaction_id'])

# Variable objetivo
y = df['fraud']

print("\nVARIABLES PREDICTORAS:")
print(X.columns)

print("\nVARIABLE OBJETIVO:")
print(y.name)

# 3. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("\nDIMENSIONES TRAIN:")
print(X_train.shape)

print("\nDIMENSIONES TEST:")
print(X_test.shape)

# 4. VERIFICAR DESBALANCE
print("\nPORCENTAJE FRAUDE EN TRAIN:")
print(y_train.value_counts(normalize=True) * 100)

print("\nPORCENTAJE FRAUDE EN TEST:")
print(y_test.value_counts(normalize=True) * 100)

# 5. VARIABLES NUMÉRICAS
num_cols = [
    'transaction_amount_usd', 'transaction_hour', 'day_of_week', 'customer_tenure_months', 'prev_transactions_7d',
    'previous_fraud_count', 'location_risk_score', 'is_international', 'distance_from_home_km']

# 6. VARIABLES CATEGÓRICAS
cat_cols = ['device_type', 'merchant_category', 'channel', 'country']

print("\nVARIABLES NUMÉRICAS:")
print(num_cols)

print("\nVARIABLES CATEGÓRICAS:")
print(cat_cols)

# 7. PREPROCESAMIENTO
preprocessor = ColumnTransformer([
    # Escalar variables numéricas
    ('num', StandardScaler(), num_cols),
    # One Hot Encoding variables categóricas
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

print("\nPREPROCESSOR CREADO CORRECTAMENTE")

# 8. CREAR PIPELINE BASE
pipeline_base = Pipeline([ ('preprocessor', preprocessor)])

print("\nPIPELINE BASE CREADO")

# 9. PROBAR TRANSFORMACIÓN
X_train_processed = pipeline_base.fit_transform(X_train)

print("\nDATOS TRANSFORMADOS CORRECTAMENTE")
print("\nDIMENSIONES DESPUÉS DEL PREPROCESAMIENTO:")
print(X_train_processed.shape)

# 10. FINALIZACIÓN
print("\nPREPROCESAMIENTO FINALIZADO")