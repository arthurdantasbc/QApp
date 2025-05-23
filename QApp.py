import streamlit as st
import numpy as np
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import RealAmplitudes
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary, LinearEqualityToPenalty
from qiskit_algorithms.utils import algorithm_globals
import warnings
import time
import sys
import math
from math import exp
from pyDOE2 import lhs
from sklearn.cluster import KMeans

parametros_treino=[
    [5.64955258, 5.13768523],
    [3.61058585, 1.50012797],
    [4.010099, 3.52868256],
    [4.11607976, 5.6834854],
    [0.06207073, 5.94693377],
    [2.95672097, 2.50510161],
    [4.5185035, 4.89354295],
    [0.3059588, 4.61665871],
    [1.16213395, 2.69425644],
    [3.2913161, 4.62269263],
    [5.39444087, 0.93767015],
    [1.65486403, 3.92155331],
    [0.9007122, 1.12241408],
    [3.5433401, 0.36532233],
    [5.44229483, 0.14221492],
    [5.59171369, 0.54184375],
    [5.77141418, 1.36856365],
    [4.88822815, 4.4179515],
    [2.21347623, 2.5046945],
    [5.59580687, 1.93161085],
    [5.43202626, 4.43408805],
    [1.74131047, 3.28836299],
    [1.11717397, 1.40617162],
    [5.10379713, 4.82242841],
    [0.94864183, 4.26102119],
    [1.22151151, 4.17421882],
    [2.48937933, 2.39064838],
    [2.43619386, 0.59423984],
    [5.94310436, 1.3699992],
    [4.63213223, 0.40957529],
    [0.75894679, 6.23837798],
    [2.77578539, 1.436039],
    [6.26838495, 1.37941869],
    [0.41929643, 0.24710771],
    [4.72602909, 2.861201],
    [5.40509589, 1.68638764],
    [0.29483925, 0.7874109],
    [2.33328555, 1.79361212],
    [5.97029726, 4.83125872],
    [3.47801, 1.46867375],
    [3.91608824, 0.71458607],
    [0.44421512, 3.37681099],
    [1.94995772, 3.18787309],
    [5.33968064, 5.06136689],
    [2.71236618, 4.98453269],
    [0.66708969, 6.00416504],
    [0.7003309, 0.18990556],
    [5.14133123, 1.89366819],
    [3.84203933, 1.56963872],
    [3.82093591, 4.77167525],
    [1.41782966, 2.12239654],
    [2.20481875, 0.74545343],
    [4.14560754, 3.93178518],
    [1.64510614, 2.99335506],
    [1.48930073, 0.68871199],
    [2.88094723, 4.14656843]
]

def generate_lhs_samples(param_intervals, num_samples):
    lhs_samples = lhs(len(param_intervals), samples=num_samples, criterion='maximin')
    lhs_scaled = np.zeros((num_samples, len(param_intervals)))

    for i in range(len(param_intervals)):
        lhs_scaled[:, i] = param_intervals[i][0] + lhs_samples[:, i] * (param_intervals[i][1] - param_intervals[i][0])

    return lhs_scaled

# Textos multilíngues
TEXTOS = {
    "pt": {
        "intro": "Seja bem-vindo ao QXplore!\nEste aplicativo foi criado para incentivar o uso da computação quântica em três áreas distintas, apresentadas a seguir.\nEscolha a área que deseja explorar e descubra as possibilidades oferecidas por essa tecnologia inovadora.",
        "pagina_otimizacao": "Otimização Quântica",
        "pagina_inferencia": "Inferência Quântica",
        "pagina_ml": "Machine Learning Quântico",
        "instancia_input": "Digite alguma coisa para testar a instância:",
        "instancia_recebida": "Instância recebida:",
        "ajuda": "Explore as possibilidades da computação quântica no idioma escolhido.",
        "idioma": "Escolha o idioma:", 
        "referencias_titulo": "Referências",
        "referencias_intro": "Para conhecer mais sobre nossos trabalhos na área, consulte as referências abaixo:"
    },
    "en": {
        "intro": "Welcome to QXplore!\nThis application was developed to promote the use of quantum computing in three distinct areas, described below.\nSelect the area you want to explore and discover the possibilities offered by this innovative technology.",
        "pagina_otimizacao": "Quantum Optimization",
        "pagina_inferencia": "Quantum Inference",
        "pagina_ml": "Quantum Machine Learning",
        "instancia_input": "Type something to test the instance:",
        "instancia_recebida": "Received instance:",
        "ajuda": "Explore the possibilities of quantum computing in the selected language.",
        "idioma": "Choose the language:", 
        "referencias_titulo": "References",
        "referencias_intro": "To learn more about our work in this area, check the references below:"
    }
}

TEXTOS_OPT = {
    "pt": {
        "insira_dados": "Insira os dados solicitados:",
        "instancia": "Instância fornecida:",
        "carregar_arquivo": "Carregar arquivo:",
        "minutos": "minutos",
        "minutos_e_segundos": "minutos e {segundos} segundos",

        # Textos da ajuda
        "problema_rap": "Problema de Alocação de Redundâncias (RAP):",
        "descricao_rap": "O RAP refere-se à otimização da alocação de componentes redundantes em um sistema para aumentar sua confiabilidade e disponibilidade.",

        "algoritmos": "Algoritmos quânticos disponíveis:",
        "descricao_algoritmos": "Os algoritmos quânticos de otimização são projetados para explorar as propriedades únicas da mecânica quântica, como superposição e entrelaçamento, para resolver problemas de otimização, como o RAP.",

        "qaoa_nome": "QAOA",
        "qaoa_desc": "Quantum Approximate Optimization Algorithm é um algoritmo quântico projetado para resolver problemas de otimização combinatória, como o RAP, aproximando-se das soluções ótimas utilizando uma sequência parametrizada de operações quânticas.",

        "vqe_nome": "VQE",
        "vqe_desc": "Variational Quantum Eigensolver é um algoritmo híbrido quântico-clássico que usa um circuito quântico variacional para encontrar o estado de menor energia de um Hamiltoniano, mas requer mais parâmetros e pode demandar mais tempo computacional em comparação com o QAOA.", 

        "modo_leitura_label": "Modo de leitura dos dados:",
        "modo_leitura_manual": "Manual",
        "modo_leitura_upload": "Upload",
        "ajuda_upload_botao": "Mostrar ajuda para upload",
        "ajuda_upload_texto": "Aqui vai o texto de ajuda para o upload de arquivos.",
        "botao_mostrar_instancia": "Mostrar instância",
        "selecionar_algoritmo": "Selecione o algoritmo:",
        "tipo_inicializacao": "Tipo de inicialização:",
        "inserir_ponto_fixo": "Insira o ponto fixo:",
        "inserir_camadas": "Número de camadas:",
        "inserir_rodadas": "Número de rodadas:", 
    },
    "en": {
        "insira_dados": "Enter the requested data:",
        "instancia": "Provided instance:",
        "carregar_arquivo": "Upload file:",
        "minutos": "minutes",
        "minutos_e_segundos": "minutes and {segundos} seconds",

        # Help section
        "problema_rap": "Redundancy Allocation Problem (RAP):",
        "descricao_rap": "RAP refers to the optimization of allocating redundant components in a system to increase its reliability and availability.",

        "algoritmos": "Available quantum algorithms:",
        "descricao_algoritmos": "Quantum optimization algorithms are designed to leverage the unique properties of quantum mechanics, such as superposition and entanglement, to solve optimization problems like RAP.",

        "qaoa_nome": "QAOA",
        "qaoa_desc": "Quantum Approximate Optimization Algorithm is a quantum algorithm designed to solve combinatorial optimization problems, such as RAP, by approximating optimal solutions using a parameterized sequence of quantum operations.",

        "vqe_nome": "VQE",
        "vqe_desc": "Variational Quantum Eigensolver is a hybrid quantum-classical algorithm that uses a variational quantum circuit to find the lowest energy state of a Hamiltonian, but it requires more parameters and may take longer computational time compared to QAOA.", 

        "modo_leitura_label": "Data input mode:",
        "modo_leitura_manual": "Manual",
        "modo_leitura_upload": "Upload",
        "ajuda_upload_botao": "Show upload help",
        "ajuda_upload_texto": "Here is the help text for uploading files.",
        "botao_mostrar_instancia": "Show instance",
        "selecionar_algoritmo": "Select the algorithm:",
        "tipo_inicializacao": "Initialization type:",
        "inserir_ponto_fixo": "Enter the fixed point:",
        "inserir_camadas": "Number of layers:",
        "inserir_rodadas": "Number of rounds:",
    }
}



def aplicar_css_botoes():
    st.markdown(
        """
        <style>
        /* Aplica estilo aos botões de forma global */
        div.stButton > button {
            background-color: #0d4376 !important;
            color: white !important;
            width: 150px !important;
            height: 80px !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            transition: background-color 0.3s ease !important;
            margin-top: 10px !important;
        }
        div.stButton > button:hover {
            background-color: #07294a !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def mostrar_introducao_e_titulo(textos):
    texto = textos['intro']
    st.markdown(
        f"""
        <div style="text-align: center; font-size:16px; color: gray; line-height:1.5;white-space: pre-line;">
            {texto}
        </div>
        <br>
        """,
        unsafe_allow_html=True
    )
    
def mostrar_referencias(textos):
    st.sidebar.markdown(f"{textos['referencias_intro']}")

    with st.sidebar.expander(textos.get("referencias_titulo", "Referências")):
        st.markdown("""
- **Araújo, L. M. M., Lins, I., Aichele, D., Maior, C., Moura, M., & Droguett, E. (2022).**  
  *Review of Quantum(-Inspired) Optimization Methods for System Reliability Problems.*  
  16th International Probabilistic Safety Assessment and Management Conference - PSAM 16.

- **Araújo, L. M. M., Lins, I., Maior, C., Aichele, D., & Droguett, E. (2022).**  
  *A Quantum Optimization Modeling for Redundancy Allocation Problems.*  
  32nd European Safety and Reliability (ESREL) Conference.

- **Araújo, L. M. M., Lins, I., Maior, C. S., Moura, M., & Droguett, E. (2023b).**  
  *A Linearization Proposal for the Redundancy Allocation Problem.*  
  INFORMS Annual Meeting.

- **Araújo, L. M. M., Raupp, L., Lins, I., & Moura, M. (2024).**  
  *Quantum Approaches for Reliability Estimation: A Systematic Literature Review.*  
  34th European Safety and Reliability (ESREL) Conference.

- **Bezerra, V., Araújo, L., Lins, I., Maior, C., & Moura, M. (2024a).**  
  *Exploring initialization strategies for quantum optimization algorithms to solve the redundancy allocation problem.*  
  34th European Safety and Reliability (ESREL) Conference.

- **Bezerra, V., Araújo, L., Lins, I., Maior, C., & Moura, M. (2024b).**  
  *Quantum optimization applied to the allocation of redundancies in systems in the Oil & Gas industry.*  
  Anais do LVI Simpósio Brasileiro de Pesquisa Operacional.

- **Bezerra, V. M. A., Araújo, L. M. M., Lins, I. D., Maior, C. B. S., & Moura, M. J. D. C. (2024).**  
  *Optimization of system reliability based on quantum algorithms considering the redundancy allocation problem.*  
  [DOI: 10.48072/2525-7579.roge.2024.3481](https://doi.org/10.48072/2525-7579.roge.2024.3481)

- **Lins, I., Araújo, L., Maior, C., Teixeira, E., Bezerra, P., Moura, M., & Droguett, E. (2023).**  
  *Quantum Optimization for Redundancy Allocation Problem Considering Various Subsystems.*  
  33th European Safety and Reliability (ESREL) Conference.
        """)

def mostrar_cartoes_de_area(textos):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("")
    with col2:
        st.image("opt3.png", width=150)
        if st.button(textos["pagina_otimizacao"], key="otimizacao_btn"):
            st.session_state['pagina'] = 'otimizacao'
    with col3:
        st.image("ml3.png", width=150)
        if st.button(textos["pagina_ml"], key="ml_btn"):
            st.session_state['pagina'] = 'ml'
    with col4:
        st.image("infer3.png", width=150)
        if st.button(textos["pagina_inferencia"], key="inferencia_btn"):
            st.session_state['pagina'] = 'inferencia'
    with col5:
        st.write("")

def ler_manualmente(textos):
    valor = st.text_input(textos["instancia_input"])
    if valor:
        return {"valor": valor}
    return None

def mostrar_instancia(instancia, textos):
    st.write(textos["instancia_recebida"])
    st.json(instancia)

def mostrar_logo_topo():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("qxplore.png", width=600)
        
#Otimização
def ler_manualmente(textos_otim):
    st.write(textos_otim["insira_dados"])
    s = st.number_input("s:", key='s', step=1)
    nj_max = st.number_input("nj_max:", key='nj_max', step=1)
    nj_min = st.number_input("nj_min:", key='nj_min', step=1)
    ctj_of = st.number_input("ctj_of:", key='ctj_of', step=1)
    
    Rjk_of, cjk_of = [], []
    for i in range(int(ctj_of)):
        Rjk_of.append(st.number_input(f"Rjk_of[{i+1}]:", key=f'Rjk_of_{i}', step=0.01, format="%.8f"))
        cjk_of.append(st.number_input(f"cjk_of[{i+1}]:", key=f'cjk_of_{i}', step=1))

    C_of = st.number_input("C_of:", key='C_of', step=1)
    return [[s, nj_max, nj_min, ctj_of, Rjk_of, cjk_of, C_of]]

def mostrar_instancia(instancia, textos_otim):
    st.subheader(textos_otim["instancia"])
    st.write("s:", instancia[0][0])
    st.write("nj_max:", instancia[0][1])
    st.write("nj_min:", instancia[0][2])
    st.write("ctj_of:", instancia[0][3])
    for i in range(int(instancia[0][3])):
        st.write(f"Rjk_of[{i+1}]:", f"{instancia[0][4][i]:.8f}")
        st.write(f"cjk_of[{i+1}]:", instancia[0][5][i])
    st.write("C_of:", instancia[0][6])

def ler_do_drive(textos_otim):
    arquivo = st.file_uploader(textos_otim["carregar_arquivo"], type=['txt'])
    if arquivo is not None:
        dados = arquivo.readlines()
        return [eval(linha.strip()) for linha in dados]
    return []

def formatar_tempo(segundos, textos_otim):
    minutos = math.floor(segundos / 60)
    segundos_restantes = math.ceil(segundos % 60)
    if segundos_restantes == 60:
        segundos_restantes = 0
        minutos += 1
    return (
        f"{minutos} {textos_otim['minutos']}"
        if segundos_restantes == 0 else
        f"{minutos} {textos_otim['minutos_e_segundos'].format(segundos=segundos_restantes)}"
    )

def mostrar_ajuda(textos_otim):

    st.sidebar.markdown(f"**{textos_otim['problema_rap']}**\n{textos_otim['descricao_rap']}")
    st.sidebar.markdown(f"**{textos_otim['algoritmos']}**\n{textos_otim['descricao_algoritmos']}")

    with st.sidebar.expander(textos_otim["qaoa_nome"]):
        st.markdown(f"**_{textos_otim['qaoa_nome']}_:** {textos_otim['qaoa_desc']}")

    with st.sidebar.expander(textos_otim["vqe_nome"]):
        st.markdown(f"**_{textos_otim['vqe_nome']}_:** {textos_otim['vqe_desc']}")


def main():
    st.set_page_config(page_title="qxplore", layout="wide")

    aplicar_css_botoes()

    # 1 - imagem no topo da sidebar
    st.sidebar.image("CM.png", use_container_width=True)

    # 2 - escolha de idioma logo abaixo da imagem
    idioma = st.sidebar.selectbox("🌐 " + TEXTOS["pt"]["idioma"], ("Português", "English"))
    lang = "pt" if idioma == "Português" else "en"
    textos = TEXTOS[lang]
    textos_otim = TEXTOS_OPT[lang]

    # 3 - aviso para clicar na imagem
    st.sidebar.info(textos["ajuda"])

    # 4 - referências em expander
    mostrar_referencias(textos)

    mostrar_logo_topo()

    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = 'inicio'

    if st.session_state['pagina'] == 'inicio':
        mostrar_introducao_e_titulo(textos)
        mostrar_cartoes_de_area(textos)

    elif st.session_state['pagina'] == 'otimizacao':
        st.subheader(textos["pagina_otimizacao"])
        
        # Aplica estilos personalizados
        st.markdown("""
            <style>
            div[role="radiogroup"] > label > div:first-child {
                background-color: #03518C;
                border-radius: 50%;
                width: 20px;
                height: 20px;
            }
            body {
                color: #000000;
                background-color: #B6D0E4;
            }
            .centered-box {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .loading-gif {
                width: 100px;
            }
            .loading-text {
                margin-top: 10px;
                font-size: 18px;
            }
            .stButton > button {
                color: #ffffff !important;
                background-color: #03518C !important;
                border-color: #03518C !important;
            }
            .stButton > button:active {
                background-color: #02416B !important;
            }
            .st-Radio > div > label {
                color: #03518C !important;
            }
            .st-Radio > div > div {
                border-color: #03518C !important;
            }
            a {
                color: #03518C !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
    
        # Ajuda
        mostrar_ajuda(textos_otim)
    
        # Leitura de dados
        modo_leitura = st.radio(
            textos_otim["modo_leitura_label"],
            (textos_otim["modo_leitura_manual"], textos_otim["modo_leitura_upload"]),
            key='modo_leitura'
        )
    
        dados = []
        if modo_leitura == textos_otim["modo_leitura_manual"]:
            dados = ler_manualmente(textos_otim)
        elif modo_leitura == textos_otim["modo_leitura_upload"]:
            if st.button(textos_otim["ajuda_upload_botao"]):
                st.markdown(textos_otim["ajuda_upload_texto"], unsafe_allow_html=True)
            dados = ler_do_drive()
    
        # Verifica se os dados estão válidos
        if (modo_leitura == textos_otim["modo_leitura_manual"] and len(dados[0]) == 7) or \
           (modo_leitura == textos_otim["modo_leitura_upload"] and dados):
    
            if st.button(textos_otim["botao_mostrar_instancia"]):
                mostrar_instancia(dados, textos_otim)
    
            if len(dados[0]) != 1:
                modo_algoritmo = st.radio(textos_otim["selecionar_algoritmo"], ('QAOA', 'VQE'))
    
                if modo_algoritmo == 'VQE':
                    tipo_inicializacao = st.radio(
                        textos_otim["tipo_inicializacao"],
                        ('LHS', 'Randômica', 'Ponto Fixo')
                    )
                    if tipo_inicializacao == 'Ponto Fixo':
                        numero_ponto_fixo = st.number_input(textos_otim["inserir_ponto_fixo"], step=0.1)
    
                elif modo_algoritmo == 'QAOA':
                    tipo_inicializacao = st.radio(
                        textos_otim["tipo_inicializacao"],
                        ('Clusterização', 'LHS', 'Randômica', 'Ponto Fixo')
                    )
                    if tipo_inicializacao == 'Ponto Fixo':
                        numero_ponto_fixo = st.number_input(textos_otim["inserir_ponto_fixo"], step=0.1)
    
                camadas = st.number_input(textos_otim["inserir_camadas"], min_value=1, max_value=3, value=1)
                rodadas = st.number_input(textos_otim["inserir_rodadas"], min_value=1, value=1)

    elif st.session_state['pagina'] == 'ml':
        st.subheader(textos["pagina_ml"])
        st.write("Conteúdo da página de Machine Learning Quântico.")

    elif st.session_state['pagina'] == 'inferencia':
        st.subheader(textos["pagina_inferencia"])
        st.write("Conteúdo da página de Inferência Quântica.")

if __name__ == "__main__":
    main()
