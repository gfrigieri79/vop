import streamlit as st
import pandas as pd
from joblib import load
import os

# --- 1. Configuração da Página ---
st.set_page_config(
    page_title="aiVOP & SAGE Estimator", 
    page_icon="🔬", 
    layout="wide"
)

# --- 2. CSS Personalizado ---
st.markdown("""
<style>
    .big-title { text-align: center; font-size: 38px !important; font-weight: 700; color: #007BFF; margin-bottom: 10px; }
    .stForm { border: 1px solid #E0E0E0; border-radius: 10px; padding: 20px; background-color: #FBFBFB; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; background-color: #007BFF; color: white; font-weight: bold; border-radius: 8px; height: 3em; }
</style>
""", unsafe_allow_html=True)

# --- 3. Função para Carregar os Modelos ---
@st.cache_resource 
def load_models():
    # Caminhos dos arquivos (Certifique-se que o nome do arquivo SAGE está correto aqui)
    path_vop = 'best_rf_model_medpwv.joblib'
    path_sage = 'best_rf_model.joblib' 
    
    models = {}
    
    if os.path.exists(path_vop):
        data_vop = load(path_vop)
        models['vop'] = {'model': data_vop['model'], 'features': data_vop['features']}
    
    if os.path.exists(path_sage):
        data_sage = load(path_sage)
        # Se o SAGE usar a chave 'best_model', o código abaixo trata isso:
        m_sage = data_sage.get('best_model', data_sage.get('model'))
        f_sage = data_sage.get('features_used', data_sage.get('features'))
        models['sage'] = {'model': m_sage, 'features': f_sage}
    else:
        st.error(f"Arquivo do modelo SAGE ({path_sage}) não encontrado!")
        
    return models

all_models = load_models()

# --- 4. Barra Lateral (Sidebar) ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #007BFF;'>VOP & SAGE</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("Este sistema utiliza IA para estimar a Velocidade da Onda de Pulso (VOP) e o escore SAGE.")
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #DC3545; font-size: 14px; font-weight: bold;'>⚠️ SISTEMA PARA TESTES</p>", unsafe_allow_html=True)

# --- 5. Área Principal ---
st.markdown("<h1 class='big-title'>Análise Cardiovascular Preditiva</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6C757D; margin-bottom: 30px;'>Insira os dados clínicos para calcular as estimativas simultâneas.</p>", unsafe_allow_html=True)

# Usamos as features do modelo VOP como base (já que são as mesmas)
if 'vop' in all_models:
    features_used = all_models['vop']['features']
    
    col_p, col_m, col_post = st.columns([1, 4, 1])

    with col_m:
        with st.form("medical_form"):
            st.markdown("##### 📝 Dados Clínicos do Paciente")
            in_col1, in_col2 = st.columns(2)
            user_inputs = {}

            for i, feature in enumerate(features_used):
                target_col = in_col1 if i % 2 == 0 else in_col2
                user_inputs[feature] = target_col.number_input(f"{feature}", value=None, placeholder="0.00", min_value=0.0)

            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Gerar Análise")

        if submit_button:
            if any(v is None for v in user_inputs.values()):
                st.warning("⚠️ Preencha todos os campos para a análise.")
            else:
                user_df = pd.DataFrame([user_inputs])
                
                # Predições
                pred_vop = all_models['vop']['model'].predict(user_df)[0]
                
                # SAGE (Verifica se o modelo carregou antes de predizer)
                pred_sage = None
                if 'sage' in all_models:
                    pred_sage = all_models['sage']['model'].predict(user_df)[0]

                st.markdown("### 📊 Resultados da Estimativa")
                
                # Criando duas colunas para mostrar os resultados lado a lado
                res_col1, res_col2 = st.columns(2)

                # --- CARD VOP ---
                with res_col1:
                    v_cor = "#D32F2F" if pred_vop > 10 else "#2E7D32"
                    v_bg = "#FFE6E6" if pred_vop > 10 else "#E8F5E9"
                    st.markdown(f"""
                        <div style="background-color: {v_bg}; padding: 20px; border-radius: 10px; border: 2px solid {v_cor}; text-align: center;">
                            <h4 style="color: {v_cor}; margin: 0;">MedPwv</h4>
                            <h1 style="color: {v_cor}; margin: 10px 0;">{pred_vop:.2f}</h1>
                            <p style="color: {v_cor}; font-size: 12px;">m/s</p>
                        </div>
                    """, unsafe_allow_html=True)

                # --- CARD SAGE ---
                with res_col2:
                    if pred_sage is not None:
                        # Lógica de cor para SAGE: Vermelho se >= 8, Verde se < 8
                        s_cor = "#D32F2F" if pred_sage >= 8 else "#2E7D32"
                        s_bg = "#FFE6E6" if pred_sage >= 8 else "#E8F5E9"
                        
                        st.markdown(f"""
                            <div style="background-color: {s_bg}; padding: 20px; border-radius: 10px; border: 2px solid {s_cor}; text-align: center;">
                                <h4 style="color: {s_cor}; margin: 0;">SAGE Estimado</h4>
                                <h1 style="color: {s_cor}; margin: 10px 0;">{pred_sage:.2f}</h1>
                                <p style="color: {s_cor}; font-size: 12px;">Escore Preditivo</p>
                            </div>
                        """, unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# from joblib import load
# import os

# # --- 1. Configuração da Página e Título Centralizado no Navegador ---
# st.set_page_config(
#     page_title="VOP Estimator", 
#     page_icon="🔬", 
#     layout="wide" # Usa a largura total da tela, fica melhor em desktops
# )

# # --- 2. CSS Personalizado para Refinar a Aparência (O pulo do gato!) ---
# st.markdown("""
# <style>
#     /* Estiliza o título principal para ser centralizado e limpo */
#     .big-title {
#         text-align: center;
#         font-size: 40px !important;
#         font-weight: 700;
#         color: #007BFF; /* Cor primária do tema */
#         margin-bottom: 20px;
#     }
#     /* Estiliza o formulário para ter uma borda suave e sombra */
#     .stForm {
#         border: 1px solid #E0E0E0;
#         border-radius: 10px;
#         padding: 20px;
#         background-color: #FBFBFB;
#         box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
#     }
#     /* Estiliza o botão para parecer mais profissional */
#     .stButton>button {
#         width: 100%;
#         background-color: #007BFF;
#         color: white;
#         font-weight: bold;
#         border-radius: 8px;
#         border: none;
#         padding: 10px;
#         transition: background-color 0.3s;
#     }
#     .stButton>button:hover {
#         background-color: #0056b3;
#         color: white;
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- 3. Carregar o Modelo ---
# @st.cache_resource 
# def load_model():
#     model_path = 'best_rf_model_medpwv.joblib'
#     if os.path.exists(model_path):
#         model_data = load(model_path)
#         return model_data['model'], model_data['features']
#     else:
#         st.error("Arquivo do modelo não encontrado!")
#         return None, None

# model, features_used = load_model()

# # ==========================================
# # --- 4. Barra Lateral (Sidebar) ---
# # Organizamos as informações institucionais aqui
# # ==========================================
# with st.sidebar:
#     st.markdown("<h2 style='text-align: center; color: #007BFF;'>aiVOP</h2>", unsafe_allow_html=True)
#     # DICA: Você pode colocar o logo da brain4care aqui se tiver o link da imagem:
#     # st.image("link_do_seu_logo.png", use_column_width=True)
    
#     st.markdown("---")
#     st.markdown("### Sobre o Projeto")
#     st.write("Este sistema utiliza Inteligência Artificial para estimar a Velocidade da Onda de Pulso (MedPwv).")
    
#     st.markdown("---")
#     # Mantemos o aviso de teste aqui, mas mais discreto e profissional
#     st.markdown("<p style='text-align: center; color: #DC3545; font-size: 14px; font-weight: bold;'>⚠️ SISTEMA PARA TESTES</p>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align: center; color: #6C757D; font-size: 12px;'>Resultados para fins de demonstração e pesquisa.</p>", unsafe_allow_html=True)

# # ==========================================
# # --- 5. Área Principal (Main Content) ---
# # Centralizamos o título e o formulário
# # ==========================================

# # Título Principal Centralizado e Estilizado pelo CSS
# st.markdown("<h1 class='big-title'>Estimativa da Velocidade da onda de Pulso</h1>", unsafe_allow_html=True)

# # Subtítulo centralizado, agora mais sutil
# st.markdown("<p style='text-align: center; color: #6C757D; margin-bottom: 30px;'>Preencha os dados clínicos abaixo para obter a estimativa baseada em IA.</p>", unsafe_allow_html=True)

# if model and features_used:
#     # Usamos colunas para centralizar o formulário na tela wide
#     col_pref, col_main, col_post = st.columns([1, 4, 1])

#     with col_main:
#         with st.form("medical_form"):
#             st.markdown("##### 📝 Dados Clínicos do Paciente")
            
#             # Criando 2 colunas dentro do formulário para os inputs
#             in_col1, in_col2 = st.columns(2)
#             user_inputs = {}

#             for i, feature in enumerate(features_used):
#                 target_col = in_col1 if i % 2 == 0 else in_col2
#                 # Adicionei min_value=0.0 para garantir dados válidos
#                 user_inputs[feature] = target_col.number_input(
#                     f"{feature}", 
#                     value=None, 
#                     placeholder="0.00",
#                     min_value=0.0
#                 )

#             st.markdown("<br>", unsafe_allow_html=True) # Espaçamento
#             submit_button = st.form_submit_button("Calcular Estimativa MedPwv")

#         # --- Lógica de Predição ---
#         if submit_button:
#             # Verifica se todos os campos foram preenchidos
#             if any(v is None for v in user_inputs.values()):
#                 st.warning("⚠️ Por favor, preencha todos os campos antes de calcular.")
#             else:
#                 user_df = pd.DataFrame([user_inputs])
#                 prediction = model.predict(user_df)[0] # Pegamos o valor numérico direto

#                 st.markdown("---")
                
#                 # --- Lógica de Alerta Condicional ---
#                 if prediction > 10:
#                     # Estilo Vermelho para Alerta (Valor > 10)
#                     cor_fundo = "#FFE6E6"  # Vermelho bem claro
#                     cor_texto = "#D32F2F"  # Vermelho forte (caracteres)
#                     border_color = "#D32F2F"
#                 else:
#                     # Estilo Verde para Normalidade (Valor <= 10)
#                     cor_fundo = "#E8F5E9"  # Verde bem claro
#                     cor_texto = "#2E7D32"  # Verde forte (caracteres)
#                     border_color = "#2E7D32"

#                 # Exibição do Resultado Estilizado
#                 st.markdown(f"""
#                     <div style="
#                         background-color: {cor_fundo}; 
#                         padding: 20px; 
#                         border-radius: 10px; 
#                         border: 2px solid {border_color};
#                         text-align: center;
#                         ">
#                         <h2 style="color: {cor_texto}; margin: 0;">Valor estimado de MedPwv:</h2>
#                         <h1 style="color: {cor_texto}; font-size: 48px; margin: 10px 0;">{prediction:.2f}</h1>
#                         <p style="color: {cor_texto}; font-size: 14px; margin: 0;">
#                             {"⚠️ ATENÇÃO: Valor acima do limite de referência." if prediction > 10 else "Dentro dos padrões esperados."}
#                         </p>
#                     </div>
#                 """, unsafe_allow_html=True)

#                 st.markdown("<p style='color: #6C757D; font-size: 12px; text-align: center; margin-top: 10px;'>Estimativa gerada por modelo Random Forest validado internamente.</p>", unsafe_allow_html=True)

#                 with st.expander("🔬 Ver detalhes técnicos da entrada"):
#                     st.dataframe(user_df, use_container_width=True)









