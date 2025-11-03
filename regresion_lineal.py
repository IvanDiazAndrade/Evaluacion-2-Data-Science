from limpio import df_laboral, df_sabado, df_domingo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#Convierte horas a minutos
def hora_a_minutos(t):
    if pd.isna(t):
        return np.nan
    return t.hour * 60 + t.minute

#Limpia los datos
def preparar(df):
    df = df.copy()
    df["Hora_num"] = df["Media_hora"].apply(hora_a_minutos)
    df = df.dropna(subset=["Hora_num", "Subidas_Promedio"])
    df["Hora_num"] = df["Hora_num"].astype(float)
    df["Subidas_Promedio"] = pd.to_numeric(df["Subidas_Promedio"], errors="coerce")
    df = df.dropna(subset=["Subidas_Promedio"])
    return df

#Prepara los subconjuntos
df_laboral = preparar(df_laboral)
df_sabado  = preparar(df_sabado)
df_domingo = preparar(df_domingo)

#Entrenamiento y evaluacion
X = df_laboral[["Hora_num"]].copy()
y = df_laboral["Subidas_Promedio"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2_test = modelo.score(X_test, y_test)

print("Intercepto:", modelo.intercept_)
print("Pendiente:", modelo.coef_[0])
print("R² (test):", r2_test)
print("MAE (test):", mae)

#Gráfico 1
plt.figure(figsize=(10,5))
plt.scatter(X["Hora_num"], y, alpha=0.35, label="Datos reales")
x_line = np.arange(0, 1441, 1)
x_line_df = pd.DataFrame({"Hora_num": x_line})
y_line = modelo.predict(x_line_df)
plt.plot(x_line, y_line, linewidth=2, label="Regresión (LABORAL)")
plt.xlabel("Hora del día")
plt.ylabel("Subidas promedio")
plt.title("Regresión lineal: Hora vs Subidas (LABORAL)")
plt.legend()
plt.grid(True)
x_ticks = np.arange(0, 1441, 30)
plt.gca().set_xticks(x_ticks)
labels = [f"{h:02d}:{m:02d}" for h in range(0,24) for m in [0,30]]
labels.append("24:00")
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Gráfico 2
plt.figure(figsize=(10,5))
x_line = np.arange(0, 1441, 1)
x_line_df = pd.DataFrame({"Hora_num": x_line})

for nombre, df_tipo in [
    ("LABORAL", df_laboral),
    ("SABADO",  df_sabado),
    ("DOMINGO", df_domingo),
]:
    if len(df_tipo) < 2:
        continue
    Xt = df_tipo[["Hora_num"]].copy()
    yt = df_tipo["Subidas_Promedio"].copy()
    Xtr, Xte, ytr, yte = train_test_split(Xt, yt, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    y_line = m.predict(x_line_df)
    plt.plot(x_line, y_line, linewidth=2, label=nombre)

plt.title("Comparación de regresión por tipo de día")
plt.xlabel("Hora del día")
plt.ylabel("Subidas promedio")
plt.grid(True)
x_ticks = np.arange(0, 1441, 30)
plt.gca().set_xticks(x_ticks)
labels = [f"{h:02d}:{m:02d}" for h in range(0,24) for m in [0,30]]
labels.append("24:00")
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
