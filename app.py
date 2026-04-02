import streamlit as st
import pandas as pd
from joblib import load
import os

# --- Configuração da Página ---
st.set_page_config(page_title="Estimativa MedPwv", page_icon="🔬")


# --- Carregar o Modelo ---
# IMPORTANTE: O arquivo 'best_rf_model_medpwv.joblib' deve estar na mesma pasta do GitHub
@st.cache_resource  # Isso evita que o modelo seja recarregado toda hora
def load_model():
    model_path = 'best_rf_model_medpwv.joblib'
    if os.path.exists(model_path):
        model_data = load(model_path)
        # Ajuste aqui conforme a estrutura que você salvou no joblib
        return model_data['model'], model_data['features']
    else:
        st.error("Arquivo do modelo não encontrado!")
        return None, None


model, features_used = load_model()

# --- Interface do Usuário ---
st.title("✨ Resultado da Estimativa de MedPwv")
st.write("Insira os dados abaixo para calcular o valor estimado.")

if model and features_used:
    # Criando colunas para o formulário não ficar muito longo
    col1, col2 = st.columns(2)
    user_inputs = {}

    with st.form("my_form"):
        for i, feature in enumerate(features_used):
            # Alterna entre coluna 1 e 2
            target_col = col1 if i % 2 == 0 else col2
            user_inputs[feature] = target_col.number_input(f"Valor para {feature}", value=0.0)

        submit_button = st.form_submit_button("Calcular Estimativa")

    # --- Lógica de Predição ---
    if submit_button:
        user_df = pd.DataFrame([user_inputs])
        prediction = model.predict(user_df)

        st.success(f"### O valor estimado de MedPwv é: **{prediction[0]:.2f}**")

        with st.expander("🔬 Ver detalhes da entrada"):
            st.write(user_df)