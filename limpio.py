import pandas as pd

# Ruta del archivo (misma carpeta)
FILE_PATH = "datos_crudos.xlsb"

# Leer la única hoja del archivo
df = pd.read_excel(FILE_PATH, engine="pyxlsb")

# Filtrar por modo METRO
df_metro = df[df["Modo"].str.upper() == "METRO"].copy()

# Convertir columna 'Media_hora' a formato hora
df_metro["Media_hora"] = pd.to_datetime(df_metro["Media_hora"], errors="coerce").dt.time

# Separar en tres DataFrames según tipo de día
df_laboral = df_metro[df_metro["Tipo_dia"].str.upper() == "LABORAL"].copy()
df_sabado = df_metro[df_metro["Tipo_dia"].str.upper() == "SABADO"].copy()
df_domingo = df_metro[df_metro["Tipo_dia"].str.upper() == "DOMINGO"].copy()

# Mostrar conteo de registros por categoría
print("Total METRO:", len(df_metro))
print("LABORAL:", len(df_laboral))
print("SABADO:", len(df_sabado))
print("DOMINGO:", len(df_domingo))
print(df_metro.head())