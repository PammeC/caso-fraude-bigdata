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
columnas_analisis = df.drop(columns=['transaction_id'])

print(columnas_analisis.describe())

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

# FUNCIÓN PARA PONER DATOS EN BARRAS VERTICALES
def agregar_etiquetas_barras(ax, formato="{:.1f}"):
    for p in ax.patches:
        altura = p.get_height()
        if altura > 0:
            ax.annotate(
                formato.format(altura),
                (p.get_x() + p.get_width() / 2, altura / 2),
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold"
            )

# FUNCIÓN PARA PONER DATOS EN BARRAS HORIZONTALES
def agregar_etiquetas_barras_h(ax, formato="{:.1f}%"):
    for p in ax.patches:
        ancho = p.get_width()
        if ancho > 0:
            ax.annotate(
                formato.format(ancho),
                (ancho / 2, p.get_y() + p.get_height() / 2),
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold"
            )

# 11. GRÁFICA 01: DISTRIBUCIÓN DE CLASES
plt.figure(figsize=(7, 5))

conteo = df['fraud'].value_counts().rename(index={0: "Normal", 1: "Fraude"})
ax = conteo.plot(kind='bar', color=['steelblue', 'crimson'])

plt.title('Distribución de Clases')
plt.xlabel('Clase')
plt.ylabel('Cantidad')
plt.xticks(rotation=0)

agregar_etiquetas_barras(ax, formato="{:.0f}")

plt.tight_layout()
plt.savefig("outputs/graficos/01_distribucion_clases.png", dpi=300, bbox_inches='tight')
plt.show()

# 12. GRÁFICA 02: DISTRIBUCIÓN DEL MONTO POR CLASE
plt.figure(figsize=(8, 5))

for label, color in [(0, 'steelblue'), (1, 'crimson')]:
    plt.hist(
        df[df['fraud'] == label]['transaction_amount_usd'],
        bins=40,
        alpha=0.6,
        color=color,
        density=True,
        label='Normal' if label == 0 else 'Fraude'
    )

plt.title('Distribución del Monto de Transacción')
plt.xlabel('Monto USD')
plt.ylabel('Densidad')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/graficos/02_distribucion_monto.png", dpi=300, bbox_inches='tight')
plt.show()

# 13. GRÁFICA 03: TASA DE FRAUDE POR CANAL
plt.figure(figsize=(8, 5))

tasa_canal = df.groupby('channel')['fraud'].mean().sort_values() * 100
ax = tasa_canal.plot(kind='barh', color='coral')

plt.title('Tasa de Fraude por Canal')
plt.xlabel('% Fraude')
plt.ylabel('Canal')

agregar_etiquetas_barras_h(ax, formato="{:.2f}%")

plt.tight_layout()
plt.savefig("outputs/graficos/03_tasa_fraude_canal.png", dpi=300, bbox_inches='tight')
plt.show()

# 14. GRÁFICA 04: TASA DE FRAUDE POR PAÍS
plt.figure(figsize=(8, 5))

tasa_pais = df.groupby('country')['fraud'].mean().sort_values(ascending=False) * 100
ax = tasa_pais.plot(kind='bar', color='tomato')

plt.title('Tasa de Fraude por País')
plt.xlabel('País')
plt.ylabel('% Fraude')
plt.xticks(rotation=0)

agregar_etiquetas_barras(ax, formato="{:.2f}%")

plt.tight_layout()
plt.savefig("outputs/graficos/04_tasa_fraude_pais.png", dpi=300, bbox_inches='tight')
plt.show()

# 15. GRÁFICA 05: TASA DE FRAUDE POR TIPO DE DISPOSITIVO
plt.figure(figsize=(8, 5))

tasa_dispositivo = df.groupby('device_type')['fraud'].mean().sort_values(ascending=False) * 100
ax = tasa_dispositivo.plot(kind='bar', color='purple')

plt.title('Tasa de Fraude por Tipo de Dispositivo')
plt.xlabel('Tipo de dispositivo')
plt.ylabel('% Fraude')
plt.xticks(rotation=0)

agregar_etiquetas_barras(ax, formato="{:.2f}%")

plt.tight_layout()
plt.savefig("outputs/graficos/05_tasa_fraude_dispositivo.png", dpi=300, bbox_inches='tight')
plt.show()

# 16. GRÁFICA 06: TASA DE FRAUDE POR HORA DEL DÍA
plt.figure(figsize=(10, 5))

fraud_hour = df.groupby('transaction_hour')['fraud'].mean() * 100

sns.lineplot(
    x=fraud_hour.index,
    y=fraud_hour.values,
    marker='o'
)

for x, y in zip(fraud_hour.index, fraud_hour.values):
    plt.text(x, y, f"{y:.1f}%", ha='center', va='bottom', fontsize=8)

plt.title('Tasa de Fraude por Hora del Día')
plt.xlabel('Hora del día')
plt.ylabel('% Fraude')
plt.xticks(range(0, 24))
plt.grid(True)

plt.tight_layout()
plt.savefig("outputs/graficos/06_tasa_fraude_hora.png", dpi=300, bbox_inches='tight')
plt.show()

# 17. GRÁFICA 07: BOXPLOT DEL MONTO POR CLASE
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x='fraud',
    y='transaction_amount_usd'
)

plt.title('Monto de Transacción según Clase')
plt.xlabel('Clase')
plt.ylabel('Monto USD')
plt.xticks([0, 1], ['Normal', 'Fraude'])

plt.tight_layout()
plt.savefig("outputs/graficos/07_boxplot_monto_clase.png", dpi=300, bbox_inches='tight')
plt.show()

# 18. GRÁFICA 08: DISTRIBUCIÓN DEL RIESGO DE UBICACIÓN POR CLASE
plt.figure(figsize=(9,5))

ax = sns.kdeplot(
    data=df,
    x='location_risk_score',
    hue='fraud',
    fill=True,
    common_norm=False
)

# Promedios por clase
media_normal = df[df['fraud'] == 0]['location_risk_score'].mean()
media_fraude = df[df['fraud'] == 1]['location_risk_score'].mean()

# Líneas verticales
plt.axvline(media_normal, color='blue', linestyle='--')
plt.axvline(media_fraude, color='orange', linestyle='--')

# Texto de medias
plt.text(
    media_normal,
    1.0,
    f'Normal\n{media_normal:.2f}',
    color='blue',
    ha='center'
)

plt.text(
    media_fraude,
    0.9,
    f'Fraude\n{media_fraude:.2f}',
    color='orange',
    ha='center'
)

plt.title('Distribución del Riesgo de Ubicación por Clase')
plt.xlabel('Location Risk Score')
plt.ylabel('Densidad')

plt.savefig(
    'outputs/graficos/08_riesgo_ubicacion_clase.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# 19. GRÁFICA 09: BOXPLOT DEL RIESGO DE UBICACIÓN
plt.figure(figsize=(7,5))

ax = sns.boxplot(
    data=df,
    x='fraud',
    y='location_risk_score'
)

# Medianas
medianas = df.groupby('fraud')['location_risk_score'].median()

# Agregar texto sobre cada caja
for i, mediana in enumerate(medianas):
    plt.text(
        i,
        mediana,
        f'{mediana:.2f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='red'
    )

plt.title('Riesgo de Ubicación según Clase')
plt.xlabel('Clase')
plt.ylabel('Location Risk Score')

plt.xticks([0,1], ['Normal', 'Fraude'])

plt.savefig(
    'outputs/graficos/09_boxplot_risk_score.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# 20. GRÁFICA 10: FRAUDES PREVIOS SEGÚN CLASE
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x='fraud',
    y='previous_fraud_count'
)

plt.title('Fraudes Previos según Clase')
plt.xlabel('Clase')
plt.ylabel('Cantidad de Fraudes Previos')
plt.xticks([0, 1], ['Normal', 'Fraude'])

plt.tight_layout()
plt.savefig("outputs/graficos/10_fraudes_previos.png", dpi=300, bbox_inches='tight')
plt.show()

# 21. GRÁFICA 11: MATRIZ DE CORRELACIÓN
plt.figure(figsize=(16, 11))

corr = df[num_cols + ['fraud']].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    center=0,
    linewidths=0.5,
    square=False,
    cbar_kws={"shrink": 0.8}
)

plt.title('Matriz de Correlación', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig("outputs/graficos/11_matriz_correlacion.png", dpi=300, bbox_inches='tight')
plt.show()

# 22. MENSAJES FINALES
print("\nEDA FINALIZADO CORRECTAMENTE")
print("Gráficos guardados en outputs/graficos/")