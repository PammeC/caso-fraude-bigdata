import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARGAR DATASET
df = pd.read_csv("data/transacciones_fraude_bigdata.csv")

# 2. VER PRIMERAS FILAS
print("\nPRIMERAS FILAS:")
print(df.head())

# 3. DIMENSIONES
print("\nDIMENSIONES:")
print(df.shape)

# 4. TIPOS DE DATOS
print("\nTIPOS DE DATOS:")
print(df.info())

# 5. VALORES FALTANTES
print("\nVALORES FALTANTES:")
print(df.isnull().sum())

# 6. ESTADÍSTICAS DESCRIPTIVAS
print("\nESTADÍSTICAS:")
print(df.describe())

# 7. DISTRIBUCIÓN DE FRAUDE
print("\nDISTRIBUCIÓN DE FRAUDE:")
print(df["fraud"].value_counts())

print("\nPORCENTAJE DE FRAUDE:")
print(df["fraud"].value_counts(normalize=True) * 100)

# 8. VARIABLES NUMÉRICAS IMPORTANTES
num_cols = [
    'transaction_amount_usd','transaction_hour','customer_tenure_months','prev_transactions_7d',
    'previous_fraud_count','location_risk_score','distance_from_home_km']

print("\nPROMEDIOS POR CLASE:")
print(df.groupby('fraud')[num_cols].mean().T)

# 9. VARIABLES CATEGÓRICAS IMPORTANTES
print("\nTASA DE FRAUDE POR VARIABLES CATEGÓRICAS:")

for col in ['device_type', 'merchant_category', 'channel', 'country']:
    print(f"\n--- {col} ---")
    print(df.groupby(col)['fraud'].mean().sort_values(ascending=False) * 100)

# 10. CREAR CARPETA DE GRÁFICOS
import os
os.makedirs("outputs/graficos", exist_ok=True)

# 11. GRÁFICOS PRINCIPALES
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# -----------------------------------------
# GRÁFICA 1: DISTRIBUCIÓN DE CLASES

df['fraud'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'crimson'])

axes[0].set_title('Distribución de Clases')
axes[0].set_xticklabels(['Normal', 'Fraude'], rotation=0)
axes[0].set_ylabel('Cantidad')

# GRÁFICA 2: MONTO DE TRANSACCIONES
for label, color in [(0, 'steelblue'), (1, 'crimson')]:
    axes[1].hist(
        df[df['fraud'] == label]['transaction_amount_usd'],
        bins=40, alpha=0.6, color=color, density=True, label='Normal' if label == 0 else 'Fraude')

axes[1].set_title('Distribución del Monto (USD)')
axes[1].set_xlabel('Monto USD')
axes[1].legend()

# GRÁFICA 3: FRAUDE POR CANAL
df.groupby('channel')['fraud'].mean().sort_values().plot(
    kind='barh', ax=axes[2], color='coral')

axes[2].set_title('Tasa de Fraude por Canal')
axes[2].set_xlabel('Probabilidad de Fraude')

# GUARDAR GRÁFICOS
plt.tight_layout()
plt.savefig( "outputs/graficos/eda_distribuciones.png", dpi=300)
plt.show()

# 12. MATRIZ DE CORRELACIÓN
plt.figure(figsize=(12, 7))
corr = df[num_cols + ['fraud']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Matriz de Correlación')
plt.savefig("outputs/graficos/matriz_correlacion.png", dpi=300)
plt.show()

# 13. MENSAJES FINALES
print("\nEDA FINALIZADO CORRECTAMENTE")
print("Gráficos guardados en outputs/graficos/")