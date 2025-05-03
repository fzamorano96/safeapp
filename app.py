import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import joblib
import pandas as pd
from folium.plugins import MarkerCluster
import gzip

# =====================
# CARGA DE DATOS Y MODELOS
# =====================

# Cargar modelo binario desde archivo comprimido
with gzip.open("modelo_binario_comprimido.pkl.gz", "rb") as f:
    modelo_bin = joblib.load(f)

# Cargar modelos restantes
modelo_tipo = joblib.load("modelo_tipo_crimen.pkl")
clases_tipo = joblib.load("clases_tipo_crimen.pkl")

# Leer CSV comprimido con separador explícito
try:
    with gzip.open("CRIME_BOSTON.comprimido.csv.gz", "rt", encoding="latin1") as f:
        data = pd.read_csv(f, sep=",", engine="python")
except Exception as e:
    st.error(f"❌ Error al leer el archivo CSV: {e}")
    st.stop()

# Mostrar columnas para debug
st.write("Columnas detectadas en el archivo CSV:")
st.write(data.columns.tolist())

# Validación de columnas esperadas
if "Lat" not in data.columns or "Long" not in data.columns:
    st.error("❌ Las columnas 'Lat' y/o 'Long' no están en el archivo.")
    st.stop()

# Eliminar registros sin coordenadas
data = data.dropna(subset=["Lat", "Long"])

# =====================
# INTERFAZ DE LA APP
# =====================
st.set_page_config(page_title="SafePath - Mapa de Crimen", layout="centered")
st.title("🗺️ SafePath: Predicción de Crimen por Ubicación")
st.markdown("Haz clic en el mapa para ver si una zona tiene alta probabilidad de crimen.")

# Mapa inicial
m = folium.Map(location=[42.36, -71.05], zoom_start=13)
m.add_child(folium.LatLngPopup())

# Mostrar mapa y capturar clic
output = st_folium(m, height=500, width=700)

if output.get("last_clicked"):
    lat = output["last_clicked"]["lat"]
    lon = output["last_clicked"]["lng"]
    coords = np.array([[lat, lon]])

    # Predicción binaria
    prob_crimen = modelo_bin.predict_proba(coords)[0][1]
    prob_text = f"🧪 Probabilidad de crimen: {prob_crimen:.2%}"

    # Crear nuevo mapa centrado en el clic
    m = folium.Map(location=[lat, lon], zoom_start=15)
    icon = folium.Icon(color="red" if prob_crimen > 0.5 else "green",
                       icon="exclamation-triangle" if prob_crimen > 0.5 else "check",
                       prefix="fa")

    folium.Marker(
        location=[lat, lon],
        popup=prob_text,
        icon=icon
    ).add_to(m)

    if prob_crimen > 0.5:
        # Predicción de tipo de crimen
        probs_tipo = modelo_tipo.predict_proba(coords)[0]
        idx = np.argmax(probs_tipo)
        tipo = clases_tipo[idx]

        st.success(f"🧠 Crimen más probable: **{tipo}** ({probs_tipo[idx]:.2%})")

        df_probs = pd.DataFrame({
            "Tipo de Crimen": clases_tipo,
            "Probabilidad": probs_tipo
        }).sort_values(by="Probabilidad", ascending=False)

        st.markdown("### 🔎 Top tipos de crimen")
        st.dataframe(df_probs.head(5))

        # Círculo de peligro
        folium.Circle(
            location=[lat, lon],
            radius=250,
            color="red",
            fill=True,
            fill_opacity=0.1
        ).add_to(m)

        # Puntos cercanos de crimen
        data["Distancia"] = np.sqrt((data["Lat"] - lat)**2 + (data["Long"] - lon)**2)
        cercanos = data.nsmallest(30, "Distancia")
        cluster = MarkerCluster().add_to(m)

        for _, row in cercanos.iterrows():
            folium.CircleMarker(
                location=[row["Lat"], row["Long"]],
                radius=3,
                color="red",
                fill=True,
                fill_opacity=0.5,
                popup=row.get("OFFENSE_CODE_GROUP", "Crimen")
            ).add_to(cluster)
    else:
        st.success("✅ Zona con baja probabilidad de crimen.")

    st.markdown(f"🔍 **{prob_text}**")
    st_folium(m, height=500, width=700)
