
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import joblib
import pandas as pd
from folium.plugins import MarkerCluster

# Cargar modelos y datos
modelo_bin = joblib.load("modelo_binario.pkl")
modelo_tipo = joblib.load("modelo_tipo_crimen.pkl")
clases_tipo = joblib.load("clases_tipo_crimen.pkl")
data = pd.read_csv("CRIME_BOSTON.csv", encoding="latin1")
data = data.dropna(subset=["Lat", "Long"])
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

    prob_crimen = modelo_bin.predict_proba(coords)[0][1]
    prob_text = f"🧪 Probabilidad de crimen: {prob_crimen:.2%}"

    # Crear nuevo mapa centrado en el punto clickeado
    m = folium.Map(location=[lat, lon], zoom_start=15)

    # Agregar marcador principal con ícono
    if prob_crimen > 0.5:
        icon = folium.Icon(color="red", icon="exclamation-triangle", prefix="fa")
    else:
        icon = folium.Icon(color="green", icon="check", prefix="fa")

    folium.Marker(
        location=[lat, lon],
        popup=prob_text,
        icon=icon
    ).add_to(m)

    # Si es zona peligrosa, mostrar tipo de crimen y cercanía
    if prob_crimen > 0.5:
        # Mostrar tipo de crimen
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

        # Dibujar círculo rojo (radio 250 m)
        folium.Circle(
            location=[lat, lon],
            radius=250,
            color="red",
            fill=True,
            fill_opacity=0.1
        ).add_to(m)

        # Agregar puntos de crimen cercanos
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
                popup=row["OFFENSE_CODE_GROUP"] if "OFFENSE_CODE_GROUP" in row else "Crimen"
            ).add_to(cluster)
    else:
        st.success("✅ Zona con baja probabilidad de crimen.")

    st.markdown(f"🔍 **{prob_text}**")
    st_folium(m, height=500, width=700)
