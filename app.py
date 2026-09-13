import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

#CONFIGURAZIONE PAGINA
st.set_page_config(
    page_title="Predizione Rischio Cardiaco",
    page_icon="🫀",
    layout="centered"
)

st.title("🫀 Sistema di Valutazione Rischio Cardiaco")
st.write(
    "Inserisci i parametri clinici del paziente per calcolare la stima di rischio tramite il modello **Random Forest**.")

st.divider()


# 2. CARICAMENTO MODELLO CON CACHING
@st.cache_resource
def carica_modello():
    return joblib.load("modello_cardio.pkl")


modello = carica_modello()

# 3. INTERFACCIA INPUT CLINICI
st.header("📋 Parametri Clinici")

col1, col2 = st.columns(2)

with col1:
    eta = st.number_input("Età (Anni)", min_value=18, max_value=100, value=50)
    sesso = st.selectbox("Sesso", ["Maschio", "Femmina"])
    pressione = st.number_input("Pressione Sanguigna a Riposo (mm Hg)", min_value=80, max_value=200, value=120)
    colesterolo = st.number_input("Colesterolo Sierico (mg/dl)", min_value=100, max_value=600, value=200)
    glicemia = st.selectbox("Glicemia a Digiuno > 120 mg/dl", ["No", "Sì"])

with col2:
    tipo_dolore = st.selectbox(
        "Tipo di Dolore al Petto",
        ["Angina Tipica (TA)", "Angina Atipica (ATA)", "Dolore Non Anginoso (NAP)", "Asintomatico (ASY)"]
    )
    ekg = st.selectbox("Risultato EKG a Riposo", ["Normale", "Anomalia ST-T", "Ipertrofia Ventricolare (LVH)"])
    frequenza_max = st.number_input("Frequenza Cardiaca Massima (BPM)", min_value=60, max_value=220, value=150)
    angina_esercizio = st.selectbox("Angina da Esercizio Fisico", ["No", "Sì"])
    oldpeak = st.number_input("Depressione ST (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    st_slope = st.selectbox("Inclinazione Segmento ST", ["Crescente (Up)", "Piatto (Flat)", "Decrescente (Down)"])

# 4. MAPPATURA INPUT (ONE-HOT ENCODING)
dati_paziente = {
    'Age': eta,
    'RestingBP': pressione,
    'Cholesterol': colesterolo,
    'FastingBS': 1 if glicemia == "Sì" else 0,
    'MaxHR': frequenza_max,
    'Oldpeak': oldpeak,
    'Sex_M': 1 if sesso == "Maschio" else 0,
    'ChestPainType_ATA': 1 if tipo_dolore == "Angina Atipica (ATA)" else 0,
    'ChestPainType_NAP': 1 if tipo_dolore == "Dolore Non Anginoso (NAP)" else 0,
    'ChestPainType_TA': 1 if tipo_dolore == "Angina Tipica (TA)" else 0,
    'RestingECG_Normal': 1 if ekg == "Normale" else 0,
    'RestingECG_ST': 1 if ekg == "Anomalia ST-T" else 0,
    'ExerciseAngina_Y': 1 if angina_esercizio == "Sì" else 0,
    'ST_Slope_Flat': 1 if st_slope == "Piatto (Flat)" else 0,
    'ST_Slope_Up': 1 if st_slope == "Crescente (Up)" else 0
}

df_input = pd.DataFrame([dati_paziente])

st.divider()

# 5. PREDIZIONE E OUTPUT CON GRAFICI INTERATTIVI PLOTLY
if st.button("🔴 Calcola Rischio Clinico", use_container_width=True):
    probabilita = modello.predict_proba(df_input)[0]
    probabilita_malato = probabilita[1] * 100
    predizione = modello.predict(df_input)[0]

    st.subheader("📊 Esito della Valutazione")

    if predizione == 1:
        st.error("⚠️ **Paziente ad ALTO RISCHIO di patologia cardiaca**")
    else:
        st.success("✅ **Paziente a BASSO RISCHIO (Sano)**")

    # --- GRAFICO 1: INDICATORE DI RISCHIO A TACHIMETRO (GAUGE CHART) ---
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilita_malato,
        number={'suffix': "%"},
        title={'text': "Livello di Rischio stimato dalla Random Forest"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},  # Verde: Basso Rischio
                {'range': [30, 60], 'color': "#f1c40f"},  # Giallo: Medio Rischio
                {'range': [60, 100], 'color': "#e74c3c"}  # Rosso: Alto Rischio
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # --- GRAFICO 2: BARRE ORIZZONTALI - FEATURE IMPORTANCE ---
    st.markdown("### 🔍 Fattori Clinici Più Influenti per il Modello")

    importanze = modello.feature_importances_
    df_imp = pd.DataFrame({
        'Caratteristica': df_input.columns,
        'Importanza (%)': importanze * 100
    }).sort_values(by='Importanza (%)', ascending=True).tail(7)  # Top 7 variabili

    fig_bar = px.bar(
        df_imp,
        x='Importanza (%)',
        y='Caratteristica',
        orientation='h',
        color='Importanza (%)',
        color_continuous_scale='Reds',
        title="Impatto delle variabili sulle decisioni dei 100 alberi"
    )
    fig_bar.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)