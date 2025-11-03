import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FILE_PATH = "datos_crudos.xlsb"

# --- Carga de datos desde el archivo ---
df = pd.read_excel(FILE_PATH, engine="pyxlsb")

# Quitamos la columna SIMIT si existe, no la vamos a usar
df = df.drop(columns=["SIMIT"], errors="ignore")

# Normalizamos texto para evitar errores de mayúsculas/minúsculas y espacios
for col in ["Modo", "Comuna", "Tipo_dia"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# Solo nos interesa el METRO y días LABORALES
df = df[(df["Modo"] == "METRO") & (df["Tipo_dia"] == "LABORAL")]

# Convertir hora en minutos desde medianoche para usarlo como variable numérica
def hora_a_minutos(valor):
    try:
        if isinstance(valor, str):
            h, m, s = map(int, valor.split(":"))
            return h*60 + m
        elif isinstance(valor, (int, float)):
            return int(valor)
        else:
            return None
    except:
        return None

df["Minuto_del_dia"] = df["Media_hora"].apply(hora_a_minutos)

# Quitamos filas con datos faltantes o subidas en cero
df = df.dropna(subset=["Minuto_del_dia", "Comuna", "Subidas_Promedio"])
df = df[df["Subidas_Promedio"] > 0]

# Verificamos que aún tengamos datos
if len(df) == 0:
    print("No hay datos disponibles después del filtrado. Revisa el archivo y los filtros.")
    exit()

# Creamos la variable objetivo: alta congestión si está en el percentil 85
umbral = df["Subidas_Promedio"].quantile(0.85)
df["Congestion_Alta"] = (df["Subidas_Promedio"] > umbral).astype(int)

# Convertimos las comunas a variables binarias (one-hot encoding)
df_model = pd.get_dummies(df[["Minuto_del_dia", "Comuna", "Congestion_Alta"]], drop_first=True)

# Separamos características y variable objetivo
X = df_model.drop("Congestion_Alta", axis=1)
Y = df_model["Congestion_Alta"]

# Dividimos en entrenamiento y prueba (80%-20%), manteniendo la proporción de congestión
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Escalamos la hora para que el K-NN funcione mejor
scaler = StandardScaler()
X_train["Minuto_del_dia"] = scaler.fit_transform(X_train[["Minuto_del_dia"]])
X_test["Minuto_del_dia"] = scaler.transform(X_test[["Minuto_del_dia"]])

# Entrenamos el modelo K-NN
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_train, Y_train)

# Evaluamos desempeño en los datos de prueba
Y_pred = modelo.predict(X_test)
mapa = {0: "No", 1: "Sí"}  # Para mostrar resultados legibles
Y_test_texto = Y_test.map(mapa)
Y_pred_texto = pd.Series(Y_pred).map(mapa)

print("Accuracy:", accuracy_score(Y_test_texto, Y_pred_texto))
cm = confusion_matrix(Y_test_texto, Y_pred_texto, labels=["No", "Sí"])
print("Matriz de Confusión:\n", pd.DataFrame(cm, index=["No", "Sí"], columns=["No", "Sí"]))
print("\nReporte:\n", classification_report(Y_test_texto, Y_pred_texto, labels=["No", "Sí"]))

# --- Función interactiva para predecir congestión según hora y comuna ---
def predecir_congestion():
    hora = input("Ingrese hora (HH:MM:SS): ")
    minuto = hora_a_minutos(hora)
    if minuto is None:
        print("Hora inválida")
        return

    # Mostramos al usuario las comunas disponibles
    comunas_validas = [c.replace("Comuna_", "") for c in X.columns if c.startswith("Comuna_")]
    print("Comunas disponibles:", ", ".join(comunas_validas))
    comuna = input("Ingrese comuna: ").strip().upper()

    # Verificamos que la comuna exista
    if f"Comuna_{comuna}" not in X.columns:
        print("Comuna no válida.")
        return

    # Creamos un registro de ejemplo para predecir
    ejemplo = {col:0 for col in X.columns}
    ejemplo["Minuto_del_dia"] = minuto
    ejemplo[f"Comuna_{comuna}"] = 1
    df_ejemplo = pd.DataFrame([ejemplo])
    
    # Predicción final
    pred = modelo.predict(df_ejemplo)
    print("Predicción de alta congestión:", "Sí" if pred[0]==1 else "No")

predecir_congestion()
