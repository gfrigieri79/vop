import streamlit as st
import pandas as pd
from joblib import load
import os

# --- 1. Configuração da Página e Título Centralizado no Navegador ---
st.set_page_config(
    page_title="VOP Estimator", 
    page_icon="🔬", 
    layout="wide" # Usa a largura total da tela, fica melhor em desktops
)

# --- 2. CSS Personalizado para Refinar a Aparência (O pulo do gato!) ---
st.markdown("""
<style>
    /* Estiliza o título principal para ser centralizado e limpo */
    .big-title {
        text-align: center;
        font-size: 40px !important;
        font-weight: 700;
        color: #007BFF; /* Cor primária do tema */
        margin-bottom: 20px;
    }
    /* Estiliza o formulário para ter uma borda suave e sombra */
    .stForm {
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        background-color: #FBFBFB;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    /* Estiliza o botão para parecer mais profissional */
    .stButton>button {
        width: 100%;
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Carregar o Modelo ---
@st.cache_resource 
def load_model():
    model_path = 'best_rf_model_medpwv.joblib'
    if os.path.exists(model_path):
        model_data = load(model_path)
        return model_data['model'], model_data['features']
    else:
        st.error("Arquivo do modelo não encontrado!")
        return None, None

model, features_used = load_model()

# ==========================================
# --- 4. Barra Lateral (Sidebar) ---
# Organizamos as informações institucionais aqui
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #007BFF;'>aiVOP</h2>", unsafe_allow_html=True)
    # DICA: Você pode colocar o logo da brain4care aqui se tiver o link da imagem:
    # st.image("link_do_seu_logo.png", use_column_width=True)
    
    st.markdown("---")
    st.markdown("### Sobre o Projeto")
    st.write("Este sistema utiliza Inteligência Artificial para estimar a Velocidade da Onda de Pulso (MedPwv).")
    
    st.markdown("---")
    # Mantemos o aviso de teste aqui, mas mais discreto e profissional
    st.markdown("<p style='text-align: center; color: #DC3545; font-size: 14px; font-weight: bold;'>⚠️ SISTEMA PARA TESTES</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6C757D; font-size: 12px;'>Resultados para fins de demonstração e pesquisa.</p>", unsafe_allow_html=True)

# ==========================================
# --- 5. Área Principal (Main Content) ---
# Centralizamos o título e o formulário
# ==========================================

# Título Principal Centralizado e Estilizado pelo CSS
st.markdown("<h1 class='big-title'>Estimativa da Velocidade da onda de Pulso</h1>", unsafe_allow_html=True)

# Subtítulo centralizado, agora mais sutil
st.markdown("<p style='text-align: center; color: #6C757D; margin-bottom: 30px;'>Preencha os dados clínicos abaixo para obter a estimativa baseada em IA.</p>", unsafe_allow_html=True)

if model and features_used:
    # Usamos colunas para centralizar o formulário na tela wide
    col_pref, col_main, col_post = st.columns([1, 4, 1])

    with col_main:
        with st.form("medical_form"):
            st.markdown("##### 📝 Dados Clínicos do Paciente")
            
            # Criando 2 colunas dentro do formulário para os inputs
            in_col1, in_col2 = st.columns(2)
            user_inputs = {}

            for i, feature in enumerate(features_used):
                target_col = in_col1 if i % 2 == 0 else in_col2
                # Adicionei min_value=0.0 para garantir dados válidos
                user_inputs[feature] = target_col.number_input(
                    f"{feature}", 
                    value=None, 
                    placeholder="0.00",
                    min_value=0.0
                )

            st.markdown("<br>", unsafe_allow_html=True) # Espaçamento
            submit_button = st.form_submit_button("Calcular Estimativa MedPwv")

        # --- Lógica de Predição ---
        if submit_button:
            # Verifica se todos os campos foram preenchidos
            if any(v is None for v in user_inputs.values()):
                st.warning("⚠️ Por favor, preencha todos os campos antes de calcular.")
            else:
                user_df = pd.DataFrame([user_inputs])
                prediction = model.predict(user_df)[0] # Pegamos o valor numérico direto

                st.markdown("---")
                
                # --- Lógica de Alerta Condicional ---
                if prediction > 10:
                    # Estilo Vermelho para Alerta (Valor > 10)
                    cor_fundo = "#FFE6E6"  # Vermelho bem claro
                    cor_texto = "#D32F2F"  # Vermelho forte (caracteres)
                    border_color = "#D32F2F"
                else:
                    # Estilo Verde para Normalidade (Valor <= 10)
                    cor_fundo = "#E8F5E9"  # Verde bem claro
                    cor_texto = "#2E7D32"  # Verde forte (caracteres)
                    border_color = "#2E7D32"

                # Exibição do Resultado Estilizado
                st.markdown(f"""
                    <div style="
                        background-color: {cor_fundo}; 
                        padding: 20px; 
                        border-radius: 10px; 
                        border: 2px solid {border_color};
                        text-align: center;
                        ">
                        <h2 style="color: {cor_texto}; margin: 0;">Valor estimado de MedPwv:</h2>
                        <h1 style="color: {cor_texto}; font-size: 48px; margin: 10px 0;">{prediction:.2f}</h1>
                        <p style="color: {cor_texto}; font-size: 14px; margin: 0;">
                            {"⚠️ ATENÇÃO: Valor acima do limite de referência." if prediction > 10 else "Dentro dos padrões esperados."}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<p style='color: #6C757D; font-size: 12px; text-align: center; margin-top: 10px;'>Estimativa gerada por modelo Random Forest validado internamente.</p>", unsafe_allow_html=True)

                with st.expander("🔬 Ver detalhes técnicos da entrada"):
                    st.dataframe(user_df, use_container_width=True)
