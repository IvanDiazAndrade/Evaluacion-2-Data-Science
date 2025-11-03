import pandas as pd
from datetime import timedelta

FILE_PATH = "datos_crudos.xlsb"
df = pd.read_excel(FILE_PATH, engine="pyxlsb")

for col in ["Modo", "Comuna", "Tipo_dia"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# Filtrar por METRO
df_metro = df[df["Modo"].str.upper() == "METRO"].copy()

#pasar a hora
def convertir_hora_excel(valor):
    try:
        return (pd.Timestamp('1899-12-30') + timedelta(days=float(valor))).time()
    except:
        return pd.NaT

df_metro["Media_hora"] = df_metro["Media_hora"].apply(convertir_hora_excel)

df_metro = df_metro[df_metro["Subidas_Promedio"] >= 1]
df_metro = df_metro.dropna(subset=["Subidas_Promedio"])



# Separar por tipo de día
df_laboral = df_metro[df_metro["Tipo_dia"].str.upper() == "LABORAL"].copy()
df_sabado = df_metro[df_metro["Tipo_dia"].str.upper() == "SABADO"].copy()
df_domingo = df_metro[df_metro["Tipo_dia"].str.upper() == "DOMINGO"].copy()



# pruebas
print("Total METRO:", len(df_metro))
print("LABORAL:", len(df_laboral))
print("SABADO:", len(df_sabado))
print("DOMINGO:", len(df_domingo))
print(df_metro.head())
