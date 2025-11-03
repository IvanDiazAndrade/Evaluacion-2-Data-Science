import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_excel("datos_crudos.xlsb", engine="pyxlsb")

df = df[df["Modo"].astype(str).str.upper().str.strip() == "METRO"].copy()
df["Subidas_Promedio"] = pd.to_numeric(df["Subidas_Promedio"], errors="coerce")
df = df.dropna(subset=["Subidas_Promedio"])
df = df[df["Subidas_Promedio"] > 0].copy()

df["Flujo"] = pd.qcut(df["Subidas_Promedio"], q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop")

features = [c for c in ["Tipo_dia", "Comuna", "Paradero", "Estacion", "Linea"] if c in df.columns]
X = pd.get_dummies(df[features], drop_first=False)
y = df["Flujo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

plt.figure(figsize=(18, 10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True, impurity=False, proportion=True, fontsize=7)
plt.title("Árbol de Decisión — Clasificación de Flujo (Metro)")
plt.tight_layout()
plt.show()

y_pred = model.predict(X_test)
print(f"\nExactitud Árbol de Decisión: {accuracy_score(y_test, y_pred):.2f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_depth=None)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Exactitud Random Forest: {accuracy_rf:.2f}")
