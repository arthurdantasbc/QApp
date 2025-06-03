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
import io
import matplotlib.pyplot as plt
import base64


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

textos_idioma = {
    "Português": "Idioma",
    "English": "Language"
}


# Textos multilíngues
TEXTOS = {
    "pt": {
        "intro": "Este aplicativo foi criado para incentivar o uso da computação quântica em três áreas distintas, apresentadas a seguir.\nEscolha a área que deseja explorar e descubra as possibilidades oferecidas por essa tecnologia inovadora.",
        "pagina_otimizacao": "Otimização Quântica",
        "pagina_inferencia": "Inferência Quântica",
        "pagina_ml": "Aprendizagem de Máquina Quântico",
        "instancia_input": "Digite alguma coisa para testar a instância:",
        "instancia_recebida": "Instância recebida:",
        "idioma": "Escolha o idioma:", 
        "referencias_titulo": "Referências",
        "referencias_intro": "Para conhecer mais sobre nossos trabalhos nas áreas, consulte as referências abaixo:",
        "info_ml": "Seção para descrever as técnicas de Machine Learning Quântico usadas.",
        "info_inf": "Seção para descrever as técnicas de Inferância Quântica usadas.",
        "titulo": "Seja bem-vindo ao <span style='color:#0d4376;'>QXplore</span>!",
         "corpo": (
             "Este aplicativo foi feito para ajudar você a conhecer e usar computação quântica em três áreas importantes.\n\n"
             "Escolha uma dessas áreas para explorar e veja como essa tecnologia pode ajudar a resolver problemas do dia a dia."
            ),
        "ini": "Página incial", 
        "pagina_referencias": "Referências"
    },
    "en": {
        "intro": "This application was developed to promote the use of quantum computing in three distinct areas, described below.\nSelect the area you want to explore and discover the possibilities offered by this innovative technology.",
        "pagina_otimizacao": "Quantum Optimization",
        "pagina_inferencia": "Quantum Inference",
        "pagina_ml": "Quantum Machine Learning",
        "instancia_input": "Type something to test the instance:",
        "instancia_recebida": "Received instance:",
        "idioma": "Choose the language:", 
        "referencias_titulo": "References",
        "referencias_intro": "To learn more about our work in this areas, check the references below:", 
        "info_ml": "Section describing the Quantum Machine Learning techniques used.",
        "info_inf": "Section describing the Quantum Inference techniques used.",
        "titulo": "Welcome to <span style='color:#0d4376;'>QXplore</span>!",
            "corpo": (
                "This application was created to help you learn and use quantum computing in three key areas.\n\n"
                "Choose one of these areas to explore and see how this technology can help solve everyday problems."
            ),
        "ini": "Homepage",
        "pagina_referencias": "References"
    }
}

TEXTOS_OPT = {
    "pt": {
        "idioma": "Idioma",
        "insira_dados": "Insira os dados do problema a ser analisado:",
        "instancia": "Instância fornecida:",
        "carregar_arquivo": "Carregar arquivo:",
        "minutos": "minutos",
        "minutos_e_segundos": "minutos e {segundos} segundos",
        # Textos da ajuda
        "problema_rap": "Problema de Alocação de Redundâncias (RAP):",
        "descricao_rap": "O RAP refere-se à otimização da alocação de componentes redundantes em um sistema para aumentar sua confiabilidade e disponibilidade.",
        "algoritmos": "Algoritmos quânticos disponíveis:",
        "inicializacoes_titulo": "Métodos de Inicialização",
        "inicializacoes_descricao": (
            "**Clusterização:** parâmetros baseados nos centros dos clusters ótimos.\n\n"
            "**LHS:** amostragem uniforme pelo hipercubo latino.\n\n"
            "**Randômica:** parâmetros iniciados aleatoriamente.\n\n"
            "**Ponto Fixo:** valores iniciais fixos e pré-definidos."),
        "descricao_algoritmos": "Os algoritmos quânticos de otimização são projetados para explorar as propriedades únicas da mecânica quântica, como superposição e entrelaçamento, para resolver problemas de otimização, como o RAP.",
        "qaoa_nome": "QAOA",
        "qaoa_desc": "Quantum Approximate Optimization Algorithm é um algoritmo quântico projetado para resolver problemas de otimização combinatória, como o RAP, aproximando-se das soluções ótimas utilizando uma sequência parametrizada de operações quânticas.",
        "vqe_nome": "VQE",
        "vqe_desc": "Variational Quantum Eigensolver é um algoritmo híbrido quântico-clássico que usa um circuito quântico variacional para encontrar o estado de menor energia de um Hamiltoniano, mas requer mais parâmetros e pode demandar mais tempo computacional em comparação com o QAOA.", 
        "modo_leitura_label": "Selecione o modo de entrada dos dados:",
        "modo_leitura_manual": "Inserção manual (preencher os dados manualmente)",
        "modo_leitura_upload": "Upload de arquivo (arquivo .txt)",
        "ajuda_upload_botao": "Mostrar ajuda para upload",
        "ajuda_upload_texto": """
        <div style="background-color: #f9f9f9; margin: 0; padding: 8px; border-radius: 3px; border: 1px solid #ddd; max-width: 850px; font-size: 14px;">
            <h4 style="color: #333; font-size: 16px; margin: 10px;">Instruções para Upload</h4>
            <p style="margin: 1px 0; font-size: 14px;">O arquivo de entrada deve ser um arquivo de texto (<code>.txt</code>), onde cada linha representa uma instância, com o seguinte formato:</p>
            <pre style="background-color: #eee; padding: 5px; border-radius: 5px; font-size: 14px; margin: 1px 0;">
    [s, nj_max, nj_min, ctj_of, Rjk_of, cjk_of, C_of]
            </pre>
            <ul style="padding-left: 24px; margin: 1px 0; font-size: 14px;">
                <li style="font-size: 14px;"><code>s</code>: Número de subsistemas;</li>
                <li style="font-size: 14px;"><code>nj_max</code>: Valor máximo dos componentes;</li>
                <li style="font-size: 14px;"><code>nj_min</code>: Valor mínimo dos componentes;</li>
                <li style="font-size: 14px;"><code>ctj_of</code>: Número de elementos em <code>Rjk_of</code> e <code>cjk_of</code>;</li>
                <li style="font-size: 14px;"><code>Rjk_of</code>: Lista de valores de confiabilidade dos componentes;</li>
                <li style="font-size: 14px;"><code>cjk_of</code>: Lista de valores de custos dos componentes;</li>
                <li style="font-size: 14px;"><code>C_of</code>: Limite de custo total;</li>
            </ul>
            <p style="margin: 1px; font-size: 14px;">Certifique-se de que o arquivo siga exatamente este formato para que os dados sejam lidos corretamente.</p>
            </ul>
            <p style="margin: 1px; font-size: 14px;">Clique no botão abaixo para baixar um arquivo de teste já formatado:</p>
        </div>
        """,
        "botao_mostrar_instancia": "Mostrar instância",
        "selecionar_algoritmo": "Selecione o algoritmo quântico:",
        "tipo_inicializacao": "Selecione o método de inicialização dos parâmetros:",
        "inserir_ponto_fixo": "Insira o ponto fixo:",
        "inserir_camadas": "Insira o número de camadas:",
        "inserir_rodadas": "Insira o número de rodadas:", 
        "executar": "Executar",
        "parametros_iniciais": "Parâmetros iniciais",
        "rodada": "Rodada",
        "camada": "Camada",
        "executando_qaoa": "Executando QAOA, por favor, aguarde...",
        "resultados": "Resultados",
        "energia_otima": "Energia Ótima",
        "confiabilidade_otima": "Confiabilidade Ótima",
        "componentes_solucao": "Componentes da Solução",
        "custo_total": "Custo Total da Solução",
        "medidas_energia": "Medidas Descritivas das Energias",
        "media_energia": "Média das Energias",
        "desvio_padrao_energia": "Desvio Padrão das Energias",
        "conteudo_pagina_ml": "Dantas",
        "conteudo_pagina_inferencia": "Lavínia",
        "tipo_inicializacao": "Tipo de inicialização",
        "inserir_ponto_fixo": "Insira o valor do ponto fixo",
        "tipos_inicializacao_vqe": ['LHS', 'Randômica', 'Ponto Fixo'],
        "tipos_inicializacao_qaoa": ['Clusterização', 'LHS', 'Randômica', 'Ponto Fixo'],
        "executando_vqe": "Executando VQE, por favor, aguarde...",
        "de": "de",
        "pagina_otimizacao": "Otimização Quântica",
        "s": "Número de subsistemas (s)",
        "nj_max": "Valor máximo dos componentes por subsistema (nj_max)",
        "nj_min": "Valor mínimo dos componentes por subsistema (nj_min)",
        "ctj_of": "Quantidade de tipos de componentes disponíveis (ctj_of)",
        "lista_componentes": "Informe a confiabilidade (Rjk_of) e o custo (cjk_of) de cada componente:",
        "confiabilidade": "Confiabilidade do componente (Rjk_of)",
        "custo": "Custo do componente (cjk_of)",
        "custo_total_limite": "Limite máximo de custo (C_of)",
        "selecionar_tipo_circuito": "Selecione o tipo de circuito VQE:",
        "real_amplitudes": "Real Amplitudes",
        "two_local": "Two Local",
        "opcoes_rotacao": ["rx", "ry", "rz"],
        "selecionar_rotacao": "Selecione as portas de rotação:",
        "opcoes_emaranhamento": ["cx", "cz", "iswap"],
        "selecionar_emaranhamento": "Selecione as portas de emaranhamento:",
        "tipo_inicializacao": "Selecione o método de inicialização:",
        "selecionar_otimizador": "Selecione o otimizador clássico:",
        "opcoes_otimizadores": ["SPSA", "COBYLA"],
        "inserir_shots": "Insira o número de shots:",
        "area_de_aplicacao": "Áreas de Aplicação:",
        "circuito_quantico": "Circuito Quântico",
        "Baixar": "Baixar arquivo",
        "download_text": "Caso deseje, faça o download do arquivo de teste exemplificado para usar ou visualizar."
    },
    "en": {
        "idioma": "Language",
        "carregar_arquivo": "Upload file:",
        "modo_leitura_label": "Select the data input mode:",
        "modo_leitura_manual": "Manual input (manually fill the data)",
        "modo_leitura_upload": "File upload (.txt file)",
        "minutos": "minutes",
        "minutos_e_segundos": "minutes and {segundos} seconds",
        "insira_dados": "Enter the problem data to be analyzed:",

        # Help section
        "problema_rap": "Redundancy Allocation Problem (RAP):",
        "descricao_rap": "RAP refers to the optimization of allocating redundant components in a system to increase its reliability and availability.",

        "algoritmos": "Available quantum algorithms:",
        "descricao_algoritmos": "Quantum optimization algorithms are designed to leverage the unique properties of quantum mechanics, such as superposition and entanglement, to solve optimization problems like RAP.",

        "qaoa_nome": "QAOA",
        "qaoa_desc": "Quantum Approximate Optimization Algorithm is a quantum algorithm designed to solve combinatorial optimization problems, such as RAP, by approximating optimal solutions using a parameterized sequence of quantum operations.",

        "vqe_nome": "VQE",
        "vqe_desc": "Variational Quantum Eigensolver is a hybrid quantum-classical algorithm that uses a variational quantum circuit to find the lowest energy state of a Hamiltonian, but it requires more parameters and may take longer computational time compared to QAOA.",
        "ajuda_upload_botao": "Show upload help",
        "ajuda_upload_texto": """
        <div style="background-color: #f9f9f9; margin: 0; padding: 8px; border-radius: 3px; border: 1px solid #ddd; max-width: 850px; font-size: 14px;">
            <h4 style="color: #333; font-size: 16px; margin: 10px;">Upload Instructions</h4>
            <p style="margin: 1px 0; font-size: 14px;">The input file should be a text file (<code>.txt</code>), where each line represents an instance in the following format:</p>
            <pre style="background-color: #eee; padding: 5px; border-radius: 5px; font-size: 14px; margin: 1px 0;">
    [s, nj_max, nj_min, ctj_of, Rjk_of, cjk_of, C_of]
            </pre>
            <ul style="padding-left: 24px; margin: 1px 0; font-size: 10px;">
                <li style="font-size: 14px;"><code>s</code>: Number of subsystems;</li>
                <li style="font-size: 14px;"><code>nj_max</code>: Maximum value of components;</li>
                <li style="font-size: 14px;"><code>nj_min</code>: Minimum value of components;</li>
                <li style="font-size: 14px;"><code>ctj_of</code>: Number of elements in <code>Rjk_of</code> and <code>cjk_of</code>;</li>
                <li style="font-size: 14px;"><code>Rjk_of</code>: List of component reliability values;</li>
                <li style="font-size: 14px;"><code>cjk_of</code>: List of component cost values;</li>
                <li style="font-size: 14px;"><code>C_of</code>: Total cost limit;</li>
            </ul>
            <p style="margin: 1px; font-size: 14px;">Make sure the file follows exactly this format so the data is read correctly.</p>
            </ul>
            <p style="margin: 1px; font-size: 14px;">Click the button below to download a pre-formatted test file:</p>
        </div>
        """,
        "botao_mostrar_instancia": "Show instance",
        "selecionar_algoritmo": "Select the quantum algorithm:",
        "tipo_inicializacao": "Select the parameter initialization method:",
        "inserir_ponto_fixo": "Enter the fixed point:",
        "inserir_camadas": "Enter the number of layers:",
        "inserir_rodadas": "Enter the number of iterations:",
        "executar": "Execute",
        "modo_leitura_upload": "Upload",
        "parametros_iniciais": "Initial parameters",
        "rodada": "Round",
        "camada": "Layer",
        "executando_qaoa": "Running QAOA, please wait...",
        "resultados": "Results",
        "energia_otima": "Optimal Energy",
        "confiabilidade_otima": "Optimal Reliability",
        "circuito_quantico": "Quantum Circuit",
        "componentes_solucao": "Solution Components",
        "custo_total": "Total Cost of the Solution",
        "medidas_energia": "Descriptive Measures of Energy",
        "media_energia": "Average Energy",
        "desvio_padrao_energia": "Standard Deviation of Energy",
        "conteudo_pagina_ml": "Dantas",
        "conteudo_pagina_inferencia": "Lavínia",
        "tipo_inicializacao": "Initialization type",
        "inserir_ponto_fixo": "Enter the fixed point value",
        "tipos_inicializacao_vqe": ['LHS', 'Random', 'Fixed Point'],
        "tipos_inicializacao_qaoa": ['Clustering', 'LHS', 'Random', 'Fixed Point'],
        "executando_vqe": "Running VQE, please wait...",
        "de": "of",
        "pagina_otimizacao": "Quantum Optimization",
        "s": "Number of subsystems (s)",
        "nj_max": "Maximum number of components per subsystem (nj_max)",
        "nj_min": "Minimum number of components per subsystem (nj_min)",
        "ctj_of": "Number of available component types (ctj_of)",
        "lista_componentes": "Enter the reliability (Rjk_of) and cost (cjk_of) for each component type:",
        "confiabilidade": "Reliability of component (Rjk_of)",
        "custo": "Cost of component (cjk_of)",
        "custo_total_limite": "Maximum total cost limit (C_of)",
        "inicializacoes_titulo": "Initialization Methods",
        "inicializacoes_descricao": (
            "**Clustering:** parameters based on centers of optimal clusters.\n\n"
            "**LHS:** uniform sampling via Latin Hypercube.\n\n"
            "**Random:** parameters initialized randomly.\n\n"
            "**Fixed Point:** fixed, predefined initial values."),
        "selecionar_tipo_circuito": "Select the type of VQE circuit:",
        "real_amplitudes": "Real Amplitudes",
        "two_local": "Two Local",

        "opcoes_rotacao": ["rx", "ry", "rz"],
        "selecionar_rotacao": "Select rotation gates:",

        "opcoes_emaranhamento": ["cx", "cz", "iswap"],
        "selecionar_emaranhamento": "Select entanglement gates:",

        "tipo_inicializacao": "Select the initialization method:",
        "selecionar_otimizador": "Select the classical optimizer:",
        "opcoes_otimizadores": ["SPSA", "COBYLA"], 
        "inserir_shots": "Enter the number of shots:",
        "area_de_aplicacao": "Areas of Application:",
        "modo_leitura_label": "Select the data input mode:",
        "modo_leitura_manual": "Manual input (enter the data manually)",
        "modo_leitura_upload": "File upload (.txt file)",
        "Baixar": "Download file",
        "download_text": "If you wish, download the sample test file to use or visualize.",
    }
}


TEXTOS_ML = {
    "pt": {
        "menu": "Menu",
        "select_dataset": "Escolha entre dados já existentes de vibração (rolamentos):",
        "select_dataset_button": "Selecione a base",
        "upload_title": "Importe dados próprios:",
        "upload_subtitle": "Faça upload da sua base de dados",
        "unsupported_file": "Formato de arquivo não suportado.",
        "upload_success": "Base de dados carregada com sucesso!",
        "dataset_preview": "Visualização da base de dados:",
        "select_features": "Selecione as features a serem extraídas da base (caso deseje)",
        "select_features_button": "Selecione as features:",
        "select_encoding": "Escolha a codificação quântica.",
        "select_encoding_method": "Escolha um método de codificação",
        "pqc_euler_rotations": "PQC: escolha a quantidade de rotações de Euler:",
        "select_quantity": "Selecione a quantidade",
        "choose_rotation_axis": "Escolha o eixo da rotação",
        "choose_first_axis": "Escolha o eixo da primeira rotação",
        "choose_second_axis": "Escolha o eixo da segunda rotação",
        "choose_third_axis": "Escolha o eixo da terceira rotação",
        "pqc_entanglement_gate": "PQC: escolha a porta de emaranhamento",
        "enter_patience": "Insira o valor da paciência:",
        "enter_epochs": "Insira o número de épocas:",
        "error_select_dataset": "Por favor, selecione um dataset.",
        "error_select_feature": "Por favor, selecione ao menos uma feature.",
        "error_select_encoding": "Por favor, selecione um método de codificação.",
        "error_select_axes": "Por favor, selecione os eixos das rotações para Angle encoding.",
        "error_loading_dataset": "Erro ao carregar o dataset.",
        "execution_started": "Execução iniciada!",
        "upload_file_types": "Por favor, envie um arquivo CSV, Excel ou Parquet."
    },
    "en": {
        "menu": "Menu",
        "select_dataset": "Choose from existing vibration data (bearings):",
        "select_dataset_button": "Select the dataset",
        "upload_title": "Upload your own data:",
        "upload_subtitle": "Upload your dataset",
        "unsupported_file": "Unsupported file format.",
        "upload_success": "Dataset loaded successfully!",
        "dataset_preview": "Dataset preview:",
        "select_features": "Select features to extract from the dataset (optional)",
        "select_features_button": "Select features:",
        "select_encoding": "Choose the quantum encoding.",
        "select_encoding_method": "Select an encoding method",
        "pqc_euler_rotations": "PQC: choose the number of Euler rotations:",
        "select_quantity": "Select quantity",
        "choose_rotation_axis": "Choose rotation axis",
        "choose_first_axis": "Choose first rotation axis",
        "choose_second_axis": "Choose second rotation axis",
        "choose_third_axis": "Choose third rotation axis",
        "pqc_entanglement_gate": "PQC: choose the entanglement gate",
        "enter_patience": "Enter patience value:",
        "enter_epochs": "Enter number of epochs:",
        "error_select_dataset": "Please select a dataset.",
        "error_select_feature": "Please select at least one feature.",
        "error_select_encoding": "Please select an encoding method.",
        "error_select_axes": "Please select rotation axes for Angle encoding.",
        "error_loading_dataset": "Error loading dataset.",
        "execution_started": "Execution started!",
        "upload_file_types": "Please upload a CSV, Excel, or Parquet file."
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
            transition: background-color 0.1s ease !important;
            margin-top: 10px !important;
        }
        div.stButton > button:hover {
            background-color: #07294a !important;
        }
        
        /* Aplica o mesmo estilo ao botão de download */
        div.stDownloadButton > button {
            background-color: #0d4376 !important;
            color: white !important;
            width: 150px !important;
            height: 80px !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            transition: background-color 0.1s ease !important;
            margin-top: 10px !important;
        }
        div.stDownloadButton > button:hover {
            background-color: #07294a !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def mostrar_introducao_e_titulo(textos):
    titulo = textos['titulo']
    corpo = textos['corpo']

    st.markdown(
        f"""
        <div style="text-align: center; max-width: 700px; margin: auto;">
            <h1 style="font-size: 32px; margin-bottom: 2px;">{titulo}</h1>
            <p style="font-size: 16px; line-height: 1.5; margin-top: 0;">
                {corpo}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
def mostrar_referencias(textos, textos_otim):
    st.title(textos.get("pagina_referencias_titulo", "Referências"))

    st.header(textos_otim["pagina_otimizacao"])
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

        if st.button(textos["pagina_referencias"], key="referencias_btn"):
            st.session_state['pagina'] = 'referencias'
            
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

    # Organizar inputs principais em 2 colunas
    col1, col2 = st.columns(2)
    with col1:
        s = st.number_input(f"{textos_otim['s']}:", step=1, min_value=1)
        nj_min = st.number_input(f"{textos_otim['nj_min']}:", step=1, min_value=0)
    with col2:
        nj_max = st.number_input(f"{textos_otim['nj_max']}:", step=1, min_value=1)
        ctj_of = st.number_input(f"{textos_otim['ctj_of']}:", step=1, min_value=1)

    st.markdown(f"**{textos_otim['lista_componentes']}**")

    Rjk_of = []
    cjk_of = []

    for i in range(int(ctj_of)):
        col_r, col_c = st.columns(2)
        with col_r:
            Rjk_of.append(
                st.number_input(f"{textos_otim['confiabilidade']} [{i+1}]:", 
                                key=f'Rjk_of_{i}', 
                                step=0.001, 
                                min_value=0.000, 
                                max_value=1.0, 
                                format="%.8f")
            )
        with col_c:
            cjk_of.append(
                st.number_input(f"{textos_otim['custo']} [{i+1}]:", 
                                key=f'cjk_of_{i}', 
                                step=1, 
                                min_value=0)
            )

    # Input final em destaque
    C_of = st.number_input(f"{textos_otim['custo_total_limite']}:", step=1, min_value=1)

    dados = [[s, nj_max, nj_min, ctj_of, Rjk_of, cjk_of, C_of]]
    return dados
    
def mostrar_instancia(instancia, textos_otim):
    st.subheader(textos_otim["instancia"])
    
    s, nj_max, nj_min, ctj_of = instancia[0][0], instancia[0][1], instancia[0][2], instancia[0][3]
    Rjk_of = instancia[0][4]
    cjk_of = instancia[0][5]
    C_of = instancia[0][6]
    
    # Dados gerais em uma linha de colunas
        
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("s", s)
    col2.metric("nj_max", nj_max)
    col3.metric("nj_min", nj_min)
    col4.metric("ctj_of", ctj_of)
    col5.metric("C_of", C_of)
    
    # Mostrar Rjk_of e cjk_of lado a lado em uma tabela organizada
    st.write("#### Valores de Rjk_of e cjk_of")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Rjk_of**")
        for i, val in enumerate(Rjk_of, 1):
            st.write(f"{i}: {val:.8f}")
            
    with col2:
        st.write("**cjk_of**")
        for i, val in enumerate(cjk_of, 1):
            st.write(f"{i}: {val}")
    
    st.markdown("---")


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

def mostrar_otim(textos_otim):
    with st.sidebar:
        st.markdown(f"#### {textos_otim['area_de_aplicacao']}")
        with st.sidebar.expander(textos_otim["pagina_otimizacao"]):        
            st.markdown(f"#### {textos_otim['problema_rap']}")
            st.markdown(f"{textos_otim['descricao_rap']}")

            st.markdown(f"#### {textos_otim['algoritmos']}")

            st.markdown(f"**QAOA**: {textos_otim['qaoa_desc']}")
            st.markdown(f"**VQE**: {textos_otim['vqe_desc']}")

            st.markdown(f"#### {textos_otim['inicializacoes_titulo']}")
            st.markdown(textos_otim['inicializacoes_descricao'])
        
def mostrar_ml(textos):
    with st.sidebar.expander(textos["pagina_ml"]):
        st.markdown(f"#### {'Coisas de Arthur'}")
        st.markdown(f"{textos['info_ml']}")

def mostrar_inf(textos):
    with st.sidebar.expander(textos["pagina_inferencia"]):
        st.markdown(f"#### {'Coisas de Lavínia'}")
        st.markdown(f"{textos['info_inf']}")

def main():
    st.set_page_config(
    page_title="QXplore",
    page_icon="pesq.png",
    layout="wide"
)

    aplicar_css_botoes()

    # 1 - imagem no topo da sidebar
    st.sidebar.image("CM.png", use_container_width=True)

    # 2 - escolha de idioma logo abaixo da imagem
    if 'lang' not in st.session_state:
        st.session_state.lang = None
    
    # Modal para escolha do idioma na primeira visita
    if st.session_state.lang is None:
        # Centraliza tudo usando markdown com CSS
        st.markdown(
            """
            <style>
                .centered {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: start;
                }
                .stButton > button {
                    width: 200px;
                    height: 50px;
                    font-size: 18px;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    
        st.markdown('<div class="centered">', unsafe_allow_html=True)
    
        mostrar_logo_topo()  # Sua função para exibir a logo
    
        st.markdown(
            """
            <div style="text-align: center;">
                <p style="font-size:36px; margin-bottom: 5px; font-weight: bold;">
                    Explore Quantum Computing with <span style="color:#0d4376;">QXplore!</span>
                </p>
                <p style="font-size:30px; margin-top: 0px; margin-bottom: 5px; font-weight: bold;">
                    Explore a Computação Quântica com <span style="color:#0d4376;">QXplore!</span>
                </p>
                <p style="font-size:18px; margin-top: 5px;">
                    Select your language to get started / Selecione seu idioma para começar:
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # Botões centralizados horizontalmente
        col1, col2, col3, col4= st.columns([1.65, 1, 1, 1.5])
        with col1:
            st.write("")
        with col2:
            if st.button("English"):
                st.session_state.lang = "en"
        
        with col3:
            if st.button("Português"):
                st.session_state.lang = "pt"


        st.markdown("</div>", unsafe_allow_html=True)
    
        # Caixa de informação sobre idioma
        st.info(
            "ℹ️ For a better experience, you can change the language anytime during navigation.\n\n"
            "ℹ️ Para uma melhor experiência, você pode alterar o idioma a qualquer momento durante a navegação."
        )

    
        st.stop()
    
    # 3 - Após escolha do idioma, sincroniza a seleção do sidebar com o idioma atual
    idioma_atual = "Português" if st.session_state.lang == "pt" else "English"
    idioma_selecionado = st.sidebar.selectbox(
        "Language / Idioma:",
        ("🇺🇸 English", "🇧🇷 Português"),
        index=0 if idioma_atual == "English"  else 1
    )

    # Atualiza o idioma no estado se o usuário mudar pelo selectbox
    if idioma_selecionado == "🇧🇷 Português" and st.session_state.lang != "pt":
        st.session_state.lang = "pt"
    elif idioma_selecionado == "🇺🇸 English" and st.session_state.lang != "en":
        st.session_state.lang = "en"

    lang = st.session_state.lang
    textos = TEXTOS[lang]
    textos_otim = TEXTOS_OPT[lang]

    # 4 - referências em expander

    mostrar_logo_topo()
    
        
    # Ajuda
    mostrar_otim(textos_otim)
    mostrar_ml(textos)
    mostrar_inf(textos)

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
    
        # Leitura de dados
       
        modo_leitura = st.radio(
            textos_otim["modo_leitura_label"],
            (textos_otim["modo_leitura_manual"], textos_otim["modo_leitura_upload"]),
            key=f"modo_leitura_{lang}"
        )
        
        dados = []
        with open("testeapp.txt", "r", encoding="utf-8") as f:
            conteudo_arquivo = f.read()
        if modo_leitura == textos_otim["modo_leitura_manual"]:
            dados = ler_manualmente(textos_otim)
        elif modo_leitura == textos_otim["modo_leitura_upload"]:
            if st.button(textos_otim["ajuda_upload_botao"]):
                st.markdown(textos_otim["ajuda_upload_texto"], unsafe_allow_html=True)

                st.download_button(
                    label=textos_otim["Baixar"],
                    data=conteudo_arquivo,
                    file_name="testeapp.txt",
                    mime="text/plain"
                )
            dados = ler_do_drive(textos_otim)
        
        # Verifica se os dados estão válidos
        if (modo_leitura == textos_otim["modo_leitura_manual"] and len(dados[0]) == 7) or \
           (modo_leitura == textos_otim["modo_leitura_upload"] and dados):
        
            if st.button(textos_otim["botao_mostrar_instancia"]):
                mostrar_instancia(dados, textos_otim)
        
            if len(dados[0]) != 1:
                col_alg, col_param = st.columns(2)
        
                with col_alg:
                    modo_algoritmo = st.radio(textos_otim["selecionar_algoritmo"], ('QAOA', 'VQE'))
        
                    if modo_algoritmo == 'VQE':
                        tipo_circuito = st.radio(
                            textos_otim["selecionar_tipo_circuito"], 
                            (textos_otim["real_amplitudes"], textos_otim["two_local"])
                        )
        
                        if tipo_circuito == textos_otim["two_local"]:
                            col_rot, col_ent = st.columns(2)
                            with col_rot:
                                rotacao_escolhida = st.multiselect(
                                    textos_otim["selecionar_rotacao"],
                                    textos_otim["opcoes_rotacao"]
                                )
                            with col_ent:
                                entanglement_escolhido = st.multiselect(
                                    textos_otim["selecionar_emaranhamento"],
                                    textos_otim["opcoes_emaranhamento"]
                                )
        
                        tipo_inicializacao = st.radio(
                            textos_otim["tipo_inicializacao"],
                            textos_otim["tipos_inicializacao_vqe"]
                        )
        
                        if tipo_inicializacao in ['Ponto Fixo', 'Fixed Point']:
                            numero_ponto_fixo = st.number_input(
                                textos_otim["inserir_ponto_fixo"], step=0.1
                            )
        
                    elif modo_algoritmo == 'QAOA':
                        tipo_inicializacao = st.radio(
                            textos_otim["tipo_inicializacao"],
                            textos_otim["tipos_inicializacao_qaoa"]
                        )
        
                        if tipo_inicializacao in ['Ponto Fixo', 'Fixed Point']:
                            numero_ponto_fixo = st.number_input(
                                textos_otim["inserir_ponto_fixo"], step=0.1
                            )
        
                with col_param:
                    otimizador = st.radio(
                        textos_otim["selecionar_otimizador"],
                        textos_otim["opcoes_otimizadores"]
                    )
                    camadas = st.number_input(
                        textos_otim["inserir_camadas"], min_value=1, max_value=3, value=1
                    )
                    rodadas = st.number_input(
                        textos_otim["inserir_rodadas"], min_value=1, value=1
                    )
                    shots = st.number_input(
                        textos_otim["inserir_shots"], min_value=100, value=1000
                    )
                
        if st.button(textos_otim['executar']):

            # Verifica o modo leitura escolhido (upload/manual)
            if modo_leitura == textos_otim['modo_leitura_upload']:
                instancia = dados[0]  # Dados do upload
            else:
                instancia = dados     # Dados da entrada manual
        
            # Extrai variáveis da instância
            s = instancia[0]
            nj_max = instancia[1]
            nj_min = instancia[2]
            ctj_of = instancia[3]
            Rjk_of = instancia[4]
            cjk_of = instancia[5]
            C_of = instancia[6]

            x = nj_max
            nmax = x

            i = 0
            b = []

            while x != 0:
                b.append(x % 2)
                x = np.floor(x / 2)
                i = i + 1
            nb = i

            ct = ctj_of
            Rjk = Rjk_of
            cjk = cjk_of
            C = C_of

            v = len(Rjk)

            qp = QuadraticProgram()

            for j in range(1, v + 1):
                for k in range(i):
                    var_name = f"b{k}{j}"
                    qp.binary_var(name=var_name)
            
            num_vars = len(qp.variables)

            linear_terms = {}

            for j in range(1, v + 1):
                for k in range(i):
                    linear_terms[f'b{k}{j}'] = np.log(1 - Rjk[j - 1]) * (2 ** (i - k - 1))

            qp.minimize(linear=linear_terms)

            constraint_terms = {}
            for j in range(1, v + 1):
                for k in range(i):
                    constraint_terms[f'b{k}{j}'] = cjk[j - 1] * (2 ** (i - k - 1))

            qp.linear_constraint(linear=constraint_terms, sense='<=', rhs=C, name='constraint_1')

            constraint_terms2 = {}
            for j in range(1, v + 1):
                for k in range(i):
                    constraint_terms2[f'b{k}{j}'] = (2 ** (i - k - 1))

            qp.linear_constraint(linear=constraint_terms2, sense='>=', rhs=1, name='constraint_2')

            constraint_terms3 = {}
            for j in range(1, v + 1):
                for k in range(i):
                    constraint_terms3[f'b{k}{j}'] = (2 ** (i - k - 1))

            qp.linear_constraint(linear=constraint_terms3, sense='<=', rhs=nmax, name='constraint_3')

            ineq2eq = InequalityToEquality()
            qp_eq = ineq2eq.convert(qp)

            int2bin = IntegerToBinary()
            qp_eq_bin = int2bin.convert(qp_eq)

            lineq2penalty = LinearEqualityToPenalty()
            qubo = lineq2penalty.convert(qp_eq_bin)

            qubits = np.array(qubo.variables)
            qubits = qubits.shape[0]

            op, offset = qubo.to_ising()
            
            if modo_algoritmo == 'QAOA':

                time_qaoa = 0
                energias = []
                parametros = []
                tempos_execucao = []
                componentes_otimos = [] 

                for i in range(rodadas):
                    for j in range(camadas):
                        if tipo_inicializacao == textos_otim["tipos_inicializacao_qaoa"][1]:  # LHS
                            param_intervals = [(0, 2*np.pi)] * 2 
                            lhs_samples = generate_lhs_samples(param_intervals, rodadas+1)
                            params = lhs_samples[i]
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_qaoa"][2]:
                            params = np.random.uniform(0, 2 * np.pi, 2)
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_qaoa"][3]:  # Ponto Fixo / Fixed Point
                            params = np.full(2, numero_ponto_fixo)
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_qaoa"][0]:
                            K = 2
                            Q = 56  

                            kmeans = KMeans(n_clusters=K)
                            cluster_labels = kmeans.fit_predict(parametros_treino)

                            random_cluster = np.random.randint(0, K)
                            cluster_indices = np.where(cluster_labels == random_cluster)[0]
                            closest_point_index = np.random.choice(cluster_indices)
                            params = parametros_treino[closest_point_index]
                                        
                        st.write("---")
                        st.write(f"{textos_otim['parametros_iniciais']} - {textos_otim['rodada']} {i+1} : {textos_otim['camada']} {j+1} = {', '.join(map(str, params))}")
                        loading_placeholder = st.empty()
                        
                        loading_placeholder.markdown(
                            f"""
                            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center;'>
                                <div class='loading-gif'>
                                    <img src='https://th.bing.com/th/id/R.4e7379292ef4b8d1945b1c3bc628d00d?rik=1iNOSJvqT0k%2bww&riu=http%3a%2f%2fbookrosabv.com.br%2fimagens%2floader.gif&ehk=OOTFpItH%2fvfYkf4YThgEExBU9BILk0f4c629HC36vTI%3d&risl=&pid=ImgRaw&r=0' 
                                    alt='Carregando...' width='100'>
                                </div>
                                <div class='loading-text' style='margin-top: 10px; font-size:18px;'>
                                    {textos_otim['executando_qaoa']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        st.markdown(f"<div class='counter'>{textos_otim['rodada']} {i + 1} / {rodadas}</div>", unsafe_allow_html=True)
                        algorithm_globals.random_seed = 10598

                        if otimizador == textos_otim["opcoes_otimizadores"][0]:  # SPSA
                            otimizador_instanciado = SPSA()
                        elif otimizador == textos_otim["opcoes_otimizadores"][1]:  # COBYLA
                            otimizador_instanciado = COBYLA()

                        sampler = Sampler(options={"shots": shots})
                        mes = QAOA(sampler=Sampler(), optimizer= otimizador_instanciado, initial_point=params)
                        meo = MinimumEigenOptimizer(min_eigen_solver=mes)

                        start = time.time()
                        qaoa_result = meo.solve(qubo)
                        end = time.time()

                        energias.append(qaoa_result.fval)
                        tempos_execucao.append(end - start)
                        componentes_otimos.append(qaoa_result.x)
                        st.write(qaoa_result)
                        
                        if i == (rodadas - 1):
                            st.subheader(textos_otim["circuito_quantico"])
                            qaoa_circuit = mes.ansatz
                        
                            fig = plt.figure(figsize=(6, 8))
                            ax = fig.add_subplot(111)
                            qaoa_circuit.draw(output='mpl', ax=ax)
                        
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig)
                        
                            # Converter a imagem em base64
                            data = base64.b64encode(buf.read()).decode("utf-8")
                        
                            # Exibir a imagem centralizada
                            st.markdown(
                                f"""
                                <div style="display: flex; justify-content: center;">
                                    <img src="data:image/png;base64,{data}" alt="QAOA" style="max-width: 100%;">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                energia_otimizada = min(energias)
                confiabilidade = 1 - math.exp(energia_otimizada)
                media_energia = np.mean(energias)
                desvio_padrao_energia = np.std(energias)

                indice_min_energia = energias.index(energia_otimizada)
                componente_otimo = componentes_otimos[indice_min_energia]

                #st.write("Configuração ótima dos componentes:")
                #st.write(componente_otimo)
                componentes_variaveis = []

                f= ct
                d= nb
                
                var_index = 0
                for m in range(1, f + 1):
                    componente = 0
                    for k in range(d):
                        var_value = componente_otimo[var_index]
                        componente += var_value * (2 ** (m - k - 1))
                        var_index += 1 
                    componentes_variaveis.append(componente)
                
                pesos = cjk
                custo_total = sum(c * p for c, p in zip(componentes_variaveis, pesos))
                componentes_formatados = [int(v) for v in componentes_variaveis]
                                                
                loading_placeholder.empty()  # Remove the loading GIF
                st.subheader(textos_otim['resultados'])
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label=textos_otim['energia_otima'], value=round(energia_otimizada, 4))
                    st.metric(label=textos_otim['confiabilidade_otima'], value=round(confiabilidade, 4))

                with col2:
                    st.metric(label=textos_otim['custo_total'], value=custo_total)
                    st.markdown(
                        f"""
                        <div>
                            <span style="font-size: 16px; font-weight: normal;">{textos_otim['componentes_solucao']}</span><br>
                            <span style="font-size: 32px; font-weight: normal; margin-left: 0;">{componentes_formatados}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.subheader(textos_otim['medidas_energia'])
                st.markdown(f"**{textos_otim['media_energia']}:** {round(media_energia, 4)}")
                st.markdown(f"**{textos_otim['desvio_padrao_energia']}:** {round(desvio_padrao_energia, 4)}")

            elif modo_algoritmo == 'VQE':
                time_vqe = 0
                energias = []
                parametros = []
                tempos_execucao = []

                for i in range(rodadas):
                    for j in range(camadas):
                        if tipo_circuito == textos_otim["real_amplitudes"]:
                            num_parametros = qubits * 2 * camadas  
                        elif tipo_circuito == textos_otim["two_local"]:
                            num_parametros = (len(rotacao_escolhida)*2) * camadas * qubits 

                        if tipo_inicializacao == textos_otim["tipos_inicializacao_vqe"][0]:  # LHS
                            param_intervals = [(0, 2 * np.pi)] * num_parametros  # Intervalo para cada parâmetro
                            lhs_samples = generate_lhs_samples(param_intervals, rodadas + 1)  # Gerando amostras LHS
                            params = lhs_samples[i]  # Selecionando a amostra correspondente à rodada
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_vqe"][1]:  # Randômica / Random
                            params = np.random.uniform(0, 2 * np.pi, num_parametros)  # Inicialização randômica
            
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_vqe"][2]:  # Ponto Fixo / Fixed Point
                            params = np.full(num_parametros, numero_ponto_fixo)  # Inicialização com valor fixo
            
                        st.write("---")
                        st.write(f"{textos_otim['parametros_iniciais']} - {textos_otim['rodada']} {i+1} : {textos_otim['camada']} {j+1} = {', '.join(map(str, params))}")
            
                        loading_placeholder = st.empty()
                        loading_placeholder.markdown(f"""
                        <div style='display: flex; flex-direction: column; align-items: center; justify-content: center;'>
                            <div class='loading-gif'>
                                <img src='https://th.bing.com/th/id/R.4e7379292ef4b8d1945b1c3bc628d00d?rik=1iNOSJvqT0k%2bww&riu=http%3a%2f%2fbookrosabv.com.br%2fimagens%2floader.gif&ehk=OOTFpItH%2fvfYkf4YThgEExBU9BILk0f4c629HC36vTI%3d&risl=&pid=ImgRaw&r=0' 
                                alt='Carregando...' width='100'>
                            </div>
                            <div class='loading-text' style='margin-top: 10px; font-size:18px;'>
                                {textos_otim['executando_vqe']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
                        st.markdown(f"<div class='counter'>{textos_otim['rodada']} {i + 1} / {rodadas}</div>", unsafe_allow_html=True)
            
                        algorithm_globals.random_seed = 10598

                        if tipo_circuito ==textos_otim["real_amplitudes"]:
                            variational_circuit = RealAmplitudes(qubits, reps=camadas)
                        elif tipo_circuito == textos_otim["two_local"]: 
                            variational_circuit = TwoLocal(qubits, rotacao_escolhida, entanglement_escolhido, reps=camadas)

                        if otimizador == textos_otim["opcoes_otimizadores"][0]:  # SPSA
                            otimizador_instanciado = SPSA()
                        elif otimizador == textos_otim["opcoes_otimizadores"][1]:  # COBYLA
                            otimizador_instanciado = COBYLA()

                        sampler = Sampler(options={"shots": shots})
                        mes = SamplingVQE(sampler=Sampler(), ansatz=variational_circuit, optimizer=otimizador_instanciado, initial_point=params)
                        meo = MinimumEigenOptimizer(min_eigen_solver=mes)
            
                        start = time.time()
                        vqe_result = meo.solve(qubo)
                        end = time.time()
            
                        energias.append(vqe_result.fval)
                        tempos_execucao.append(end - start)
                        componentes_otimos.append(vqe_result.x)

                        if i == (rodadas-1): 
                            st.subheader(textos_otim["circuito_quantico"])
                            vqe_circuit = mes.ansatz
                            fig = plt.figure(figsize=(6, 8)) 
                            ax = fig.add_subplot(111)
                            vqe_circuit.draw(output='mpl', ax=ax)
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            st.image(buf, caption="VQE",  use_container_width=False)
                            plt.close(fig)
            
                energia_otimizada = min(energias)
                confiabilidade = 1 - math.exp(energia_otimizada)
                media_energia = np.mean(energias)
                desvio_padrao_energia = np.std(energias)
            
                indice_min_energia = energias.index(energia_otimizada)
                componente_otimo = componentes_otimos[indice_min_energia]
            
                componentes_variaveis = []
                f = ct
                d = nb
            
                var_index = 0
                for m in range(1, f + 1):
                    componente = 0
                    for k in range(d):
                        var_value = componente_otimo[var_index]
                        componente += var_value * (2 ** (m - k - 1))
                        var_index += 1 
                    componentes_variaveis.append(componente)
            
                pesos = cjk
                custo_total = sum(c * p for c, p in zip(componentes_variaveis, pesos))
                componentes_formatados = [int(v) for v in componentes_variaveis]
            
                loading_placeholder.empty()
                st.subheader(textos_otim['resultados'])
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label=textos_otim['energia_otima'], value=round(energia_otimizada, 4))
                    st.metric(label=textos_otim['confiabilidade_otima'], value=round(confiabilidade, 4))

                with col2:
                    st.metric(label=textos_otim['custo_total'], value=custo_total)
                    st.markdown(
                        f"""
                        <div>
                            <span>{textos_otim['componentes_solucao']}</span><br>
                            <span style="font-size: 32px; font-weight: normal; margin-left: 0;">{componentes_formatados}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )       
                st.subheader(textos_otim['medidas_energia'])
                st.markdown(f"**{textos_otim['media_energia']}:** {round(media_energia, 4)}")
                st.markdown(f"**{textos_otim['desvio_padrao_energia']}:** {round(desvio_padrao_energia, 4)}")
                
        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'

    elif st.session_state['pagina'] == 'ml':
        st.subheader(textos["pagina_ml"])
        st.write("Arthur")

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'
    

    elif st.session_state['pagina'] == 'inferencia':
        st.subheader(textos["pagina_inferencia"])
        st.write("Lavínia")

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'

    elif st.session_state['pagina'] == 'referencias':
        mostrar_referencias(textos, textos_otim)

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'

if __name__ == "__main__":
    main()
