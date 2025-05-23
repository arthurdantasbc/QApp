import streamlit as st
import numpy as np
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import RealAmplitudes, TwoLocal
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
import matplotlib.pyplot as plt

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

def ler_manualmente():
    st.write("Insira os dados solicitados:")
    s = st.number_input("Número de subsistemas (s):", key='s', step=1, min_value=0)
    nj_max = st.number_input("Valor máximo dos componentes (nj_max):", key='nj_max', step=1, min_value=0)
    nj_min = st.number_input("Valor mínimo dos componentes(nj_min):", key='nj_min', step=1, min_value=0)
    ctj_of = st.number_input("Número de elementos redundantes (ctj_of):", key='ctj_of', step=1, min_value=0)

    st.write("Lista de valores de confiabilidade dos componentes (Rjk_of)")
    st.write("Lista de valores de custos dos componentes (cjk_of)")
    
    # Inicializa Rjk_of e cjk_of como listas vazias
    Rjk_of = []
    cjk_of = []
    
    # Preenche Rjk_of e cjk_of com os valores fornecidos
    for i in range(int(ctj_of)):
        Rjk_of.append(st.number_input(f"Rjk_of[{i+1}]:", key=f'Rjk_of_{i}', step=0.001, min_value=0.000))
        cjk_of.append(st.number_input(f"cjk_of[{i+1}]:", key=f'cjk_of_{i}', step=1, min_value=0))
    
    C_of = st.number_input("Limite de custo total (C_of):", key='C_of', step=1, min_value=0)

    vet = [[s, nj_max, nj_min, ctj_of, Rjk_of, cjk_of, C_of]]
    
    return vet

def mostrar_instancia(instancia):
    st.subheader("Instância fornecida:")
    st.write("s:", instancia[0][0])
    st.write("nj_max:", instancia[0][1])
    st.write("nj_min:", instancia[0][2])
    st.write("ctj_of:", instancia[0][3])
    for i in range(int(instancia[0][3])):
        st.write(f"Rjk_of[{i+1}]:", instancia[0][4][i])
        st.write(f"cjk_of[{i+1}]:", instancia[0][5][i])
    st.write("C_of:", instancia[0][6])
    st.markdown("</div>", unsafe_allow_html=True)

def ler_do_drive():
    arquivo = st.file_uploader("Carregar arquivo:", type=['txt'])
    if arquivo is not None:
        dados = arquivo.readlines()
        return [eval(linha.strip()) for linha in dados]
    else:
        return []

def formatar_tempo(segundos):
    minutos = math.floor(segundos / 60)
    segundos_restantes = math.ceil(segundos % 60)
    if segundos_restantes == 60:
        segundos_restantes = 0
        minutos += 1
    if segundos_restantes == 0:
        return f"{minutos} minutos"
    else:
        return f"{minutos} minutos e {segundos_restantes} segundos"
        
def mostrar_ajuda():
    st.sidebar.image(r'CM.png', use_container_width=True)
    st.sidebar.image(r'MA.png', use_container_width=True)
    st.sidebar.markdown("""

        **Problema de Alocação de Redundâncias (RAP):**
       O RAP refere-se à otimização da alocação de componentes redundantes em um sistema para aumentar sua confiabilidade e disponibilidade. 

    """)

    with st.sidebar.expander("Algoritmos quânticos disponíveis:"):
        st.markdown("""
        **_Quantum Approximate Optimization Algorithm (QAOA)_:**  
        É um algoritmo quântico projetado para resolver problemas de otimização combinatória, como o RAP, aproximando-se das soluções ótimas utilizando uma sequência parametrizada de operações quânticas.
    
        **_Variational Quantum Eigensolver (VQE)_:**  
        É um algoritmo híbrido quântico-clássico que usa um circuito quântico variacional para encontrar o estado de menor energia de um Hamiltoniano, mas requer mais parâmetros e pode demandar mais tempo computacional em comparação com o QAOA.
    """)


    with st.sidebar.expander("Inicializações dos parâmetros iniciais:"):
        st.markdown("""
        **_Clusterização:_**  
        Inicializa os parâmetros definindo os centros iniciais dos clusters, geralmente de forma aleatória ou com base em algum critério específico, para iniciar o agrupamento dos dados.
    
        **_LHS (Latin Hypercube Sampling):_**  
        Inicializa os parâmetros amostrando de forma uniforme o espaço das variáveis, garantindo que todas as áreas do espaço de busca sejam representadas.
    
        **_Randômica:_**  
        Inicializa os parâmetros de maneira aleatória, explorando uma ampla gama de soluções possíveis sem depender de um ponto inicial específico.
    
        **_Método do Ponto Fixo:_**  
        Inicializa os parâmetros com um valor inicial e o ajusta iterativamente até que o sistema atinja um ponto de convergência.
    """)

    with st.sidebar.expander("Referências"):
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
          Anais Do LVI Simpósio Brasileiro de Pesquisa Operacional.
    
        - **Bezerra, V. M. A., Araújo, L. M. M., Lins, I. D., Maior, C. B. S., & Moura, M. J. D. C. (2024).**  
          *Optimization of system reliability based on quantum algorithms considering the redundancy allocation problem.*  
          [DOI: 10.48072/2525-7579.roge.2024.3481](https://doi.org/10.48072/2525-7579.roge.2024.3481)
        
        - **Lins, I., Araújo, L., Maior, C., Teixeira, E., Bezerra, P., Moura, M., & Droguett, E. (2023).**  
          *Quantum Optimization for Redundancy Allocation Problem Considering Various Subsystems.*  
          33th European Safety and Reliability (ESREL) Conference.
    """)



def main():
    st.markdown("""
    <style>
    /* Muda a cor da seleção do st.radio */
    div[role="radiogroup"] > label > div:first-child {
        background-color: #03518C; /* Cor verde */
        border-radius: 50%;
        width: 20px;
        height: 20px;
    }
     </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        body {
            color: #000000;  /* Cor do texto */
            background-color: #B6D0E4;  /* Cor de fundo azul claro */
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
        /* Estilo para botões */
        .stButton > button {
            color: #ffffff !important;  /* Cor do texto no botão */
            background-color: #03518C !important;  /* Cor de fundo azul */
            border-color: #03518C !important;  /* Cor da borda azul */
        }
        /* Estilo para botão selecionado */
        .stButton > button:active {
            background-color: #02416B !important;  /* Cor de fundo azul mais escura para botão selecionado */
        }
        /* Estilo para radio buttons */
        .st-Radio > div > label {
            color: #03518C !important;  /* Cor do texto do label */
        }
        .st-Radio > div > div {
            border-color: #03518C !important;  /* Cor da borda do radio button */
        }
        .st-Radio > div > div > label::before {
            background-color: #ffffff !important;  /* Cor de fundo inicial do radio button */
            border-color: #03518C !important;  /* Cor da borda inicial do radio button */
        }
        .st-Radio > div > div > input:checked + label::before {
            background-color: #03518C !important;  /* Cor de fundo do radio button selecionado */
            border-color: #03518C !important;  /* Cor da borda do radio button selecionado */
        }
        .st-Radio > div > div > input:checked + label::after {
            background-color: #03518C !important;  /* Cor do círculo dentro do radio button selecionado */
        }
        /* Estilo para links */
        a {
            color: #03518C !important;  /* Cor do texto do link */
        }
        </style>
    """, unsafe_allow_html=True)

    st.image(r'MB.png', use_container_width=True)
    
    mostrar_ajuda()

    # Leitura dos dados
    modo_leitura = st.radio("Selecione o modo de leitura da instância:", ('Manual (dados inseridos manualmente pelo usuário)', 'Upload de arquivo (inserir arquivo txt)'), key='modo_leitura')

    dados = []
    if modo_leitura == 'Manual (dados inseridos manualmente pelo usuário)':
        dados = ler_manualmente()
    elif modo_leitura == 'Upload de arquivo (inserir arquivo txt)':
        if st.button('Ajuda'):
            # Exibir uma caixinha centralizada com uma pequena margem antes do texto
            st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; padding-top: 0px;">
                <div style="background-color: #f9f9f9; margin: 0; padding: 8px; border-radius: 3px; border: 1px solid #ddd; max-width: 500px; font-size: 10px;">
                    <h4 style="color: #333; font-size: 14px; margin: 10px ;">Instruções para Upload</h4>
                    <p style="margin: 1px 0; font-size: 12px;">O arquivo de entrada deve ser um arquivo de texto (<code>.txt</code>), onde cada linha representa uma instância, com o seguinte formato:</p>
                    <pre style="background-color: #eee; padding: 5px; border-radius: 5px; font-size: 12px; margin: 1px 0;">
            [s, nj_max, nj_min, ctj_of, Rjk_of, cjk_of, C_of]
                    </pre>
                    <ul style="padding-left: 20px; margin: 1px 0; font-size: 10px;">
                        <li style="font-size: 12px;"><code>s</code>: Número de subsistemas;</li>
                        <li style="font-size: 12px;"><code>nj_max</code>: Valor máximo dos componentes;</li>
                        <li style="font-size: 12px;"><code>nj_min</code>: Valor mínimo dos componentes;</li>
                        <li style="font-size: 12px;"><code>ctj_of</code>: Número de elementos em <code>Rjk_of</code> e <code>cjk_of</code>;</li>
                        <li style="font-size: 12px;"><code>Rjk_of</code>: Lista de valores de confiabilidade dos componentes;</li>
                        <li style="font-size: 12px;"><code>cjk_of</code>: Lista de valores de custos dos componentes;</li>
                        <li style="font-size: 12px;"><code>C_of</code>: Limite de custo total;</li>
                    </ul>
                    <p style="margin: 1px ; font-size: 12px;">Certifique-se de que o arquivo siga exatamente este formato para que os dados sejam lidos corretamente.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        dados = ler_do_drive()

    if (modo_leitura == 'Manual (dados inseridos manualmente pelo usuário)' and len(dados[0]) == 7) or (modo_leitura == 'Upload de arquivo (inserir arquivo txt)' and  len(dados) > 0):
        if st.button('Mostrar instância'):
            mostrar_instancia(dados)

        # Seleção do algoritmo
        if len(dados[0]) != 1:
            modo_algoritmo = st.radio("Selecione o algoritmo desejado:", ('QAOA', 'VQE'))

            # Inicialização dos parâmetros
            if  modo_algoritmo == 'VQE':
                tipo_circuito = st.radio("Selecione o tipo de circuito VQE:", ('Real Amplitudes', 'Two Local'))
        
                if tipo_circuito == 'Two Local':
                    # Escolha das portas de rotação
                    opcoes_rotacao = ['rx', 'ry', 'rz']
                    rotacao_escolhida = st.multiselect("Selecione as portas de rotação:", opcoes_rotacao)
            
                    # Escolha das portas de emaranhamento
                    opcoes_entanglement = ['cx', 'cz', 'iswap']
                    entanglement_escolhido = st.multiselect("Selecione as portas de emaranhamento:", opcoes_entanglement)
                    
                tipo_inicializacao = st.radio("Selecione o tipo de inicialização dos parâmetros iniciais:", ('LHS', 'Randômica', 'Ponto Fixo'))
                if tipo_inicializacao == 'Ponto Fixo':
                    numero_ponto_fixo = st.number_input("Insira o ponto fixo:", step=0.1)

            if  modo_algoritmo == 'QAOA':
                tipo_inicializacao = st.radio("Selecione o tipo de inicialização dos parâmetros iniciais:", ('Clusterização', 'LHS', 'Randômica', 'Ponto Fixo'))
                if tipo_inicializacao == 'Ponto Fixo':
                    numero_ponto_fixo = st.number_input("Insira o ponto fixo:", step=0.1)
            
        #Selcionar o otimizador
            otimizador = st.radio("Selecione o otimizador:", ('SPSA', 'COBYLA'))
            
        #Solicitar o numero de camadas
            camadas  = st.number_input("Insira o número de camadas:", min_value=1, max_value=3, value=1)

        # Solicitar a quantidade de rodadas
            rodadas = st.number_input("Insira o número de execuções:", min_value=1, value=1)

        # Solicitar número de shots
            shots = st.number_input("Insira o número de shots:", min_value=100, value=1000)

        if st.button('Executar'):

            if modo_leitura == 'Upload de arquivo (inserir arquivo txt)':
                instancia = dados[0]
            else:
                instancia = dados[0]
                    
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
            
            if modo_algoritmo == 'QAOA':

                time_qaoa = 0
                energias = []
                parametros = []
                tempos_execucao = []
                componentes_otimos = [] 

                for i in range(rodadas):
                    for j in range(camadas):
                        if tipo_inicializacao == 'LHS':
                            param_intervals = [(0, 2*np.pi)] * 2 
                            lhs_samples = generate_lhs_samples(param_intervals, rodadas+1)
                            params = lhs_samples[i]
                        elif tipo_inicializacao == 'Randômica':
                            params = np.random.uniform(0, 2 * np.pi, 2)
                        elif tipo_inicializacao == 'Ponto Fixo':
                            params = np.full(2, numero_ponto_fixo)
                        elif tipo_inicializacao == 'Clusterização':
                            K = 2
                            Q = 56  

                            kmeans = KMeans(n_clusters=K)
                            cluster_labels = kmeans.fit_predict(parametros_treino)

                            random_cluster = np.random.randint(0, K)
                            cluster_indices = np.where(cluster_labels == random_cluster)[0]
                            closest_point_index = np.random.choice(cluster_indices)
                            params = parametros_treino[closest_point_index]

                        
                        st.write("---")
                        st.write(f"Parâmetros iniciais - Rodada {i+1} : Camada {j+1} = {', '.join(map(str, params))}")

                        loading_placeholder = st.empty()  
                        loading_placeholder.markdown("""
                        <div class='loading-gif'><img src='https://th.bing.com/th/id/R.4e7379292ef4b8d1945b1c3bc628d00d?rik=1iNOSJvqT0k%2bww&riu=http%3a%2f%2fbookrosabv.com.br%2fimagens%2floader.gif&ehk=OOTFpItH%2fvfYkf4YThgEExBU9BILk0f4c629HC36vTI%3d&risl=&pid=ImgRaw&r=0' alt='Carregando...'></div>
                        <div class='loading-text'>Executando QAOA, por favor, aguarde...</div>
                        """, unsafe_allow_html=True)

                        algorithm_globals.random_seed = 10598

                        if otimizador == "SPSA":
                            otimizador_instanciado = SPSA()
                        else: 
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

                # Processamento dos resultados
                energia_otimizada = min(energias)
                confiabilidade = 1 - math.exp(energia_otimizada)
                media_energia = np.mean(energias)
                desvio_padrao_energia = np.std(energias)
                tempo_medio = np.mean(tempos_execucao)
                
                indice_min_energia = energias.index(energia_otimizada)
                componente_otimo = componentes_otimos[indice_min_energia]
                
                # Decodifica os componentes escolhidos
                componentes_variaveis = []
                var_index = 0
                for m in range(1, ct + 1):
                    componente = 0
                    for k in range(nb):
                        var_value = componente_otimo[var_index]
                        componente += var_value * (2 ** (m - k - 1))
                        var_index += 1 
                    componentes_variaveis.append(componente)
                
                # Cálculo do custo total
                custo_total = sum(c * p for c, p in zip(componentes_variaveis, cjk))
                
                # Remove o GIF de carregamento
                loading_placeholder.empty()
                
                # Exibir resultados
                st.write("---")
                st.subheader("Resultados")
                st.write(f"**Energia Ótima:** {energia_otimizada}")
                st.write(f"**Confiabilidade Ótima:** {confiabilidade:.8f}")
                st.write(f"**Componentes da Solução:** {componentes_variaveis}")
                st.write(f"**Custo Total da Solução:** {custo_total}")
                
                st.subheader("Medidas Descritivas das Energias")
                st.write(f"**Média das Energias:** {media_energia}")
                st.write(f"**Desvio Padrão das Energias:** {desvio_padrao_energia}")
                st.write(f"**Tempo Médio de Execução:** {tempo_medio:.4f} segundos")

                st.subheader("Circuito")
                fig = mes.ansatz.decompose().draw("mpl")
                st.pyplot(fig)

                # Exibir todas as energias e tempos de execução
                st.subheader("Energias e Tempos por Execução")
                for i in range(rodadas):
                    st.write(f"Rodada {i+1}: Energia = {energias[i]}, Tempo = {tempos_execucao[i]:.4f} segundos")
                
                # Exibir o circuito do QAOA
                st.subheader("Circuito do QAOA")
                exibir_circuito_qaoa(mes)

            elif modo_algoritmo == 'VQE':

                time_vqe = 0
                energias = []
                parametros = []
                tempos_execucao = []

            for i in range(rodadas):
                for j in range(camadas):
                    if tipo_circuito == 'Real Amplitudes':
                        num_parametros = qubits * 2 * camadas  
                    elif tipo_circuito == 'Two Local':
                        num_parametros = (len(rotacao_escolhida)*2) * camadas * qubits 
            
                    # Inicializando os parâmetros
                    if tipo_inicializacao == 'LHS':
                        param_intervals = [(0, 2 * np.pi)] * num_parametros  # Intervalo para cada parâmetro
                        lhs_samples = generate_lhs_samples(param_intervals, rodadas + 1)  # Gerando amostras LHS
                        params = lhs_samples[i]  # Selecionando a amostra correspondente à rodada
                    elif tipo_inicializacao == 'Randômica':
                        params = np.random.uniform(0, 2 * np.pi, num_parametros)  # Inicialização randômica
                    elif tipo_inicializacao == 'Ponto Fixo':
                        params = np.full(num_parametros, numero_ponto_fixo)  # Inicialização com valor fixo
            
                    st.write("---")
                    try:
                        st.write(f"Parâmetros iniciais - Rodada {i+1} : Camada {j+1} = {', '.join(map(str, params))}")
                    except Exception as e:
                        st.write(f"Erro: {e}")
            
                    # Mostrando indicador de carregamento enquanto o VQE é executado
                    loading_placeholder = st.empty() 
                    loading_placeholder.markdown("""
                    <div class='loading-gif'><img src='https://th.bing.com/th/id/R.4e7379292ef4b8d1945b1c3bc628d00d?rik=1iNOSJvqT0k%2bww&riu=http%3a%2f%2fbookrosabv.com.br%2fimagens%2floader.gif&ehk=OOTFpItH%2fvfYkf4YThgEExBU9BILk0f4c629HC36vTI%3d&risl=&pid=ImgRaw&r=0' alt='Carregando...'></div>
                    <div class='loading-text'>Executando VQE, por favor, aguarde...</div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrando o progresso da rodada
                    st.markdown(f"<div class='counter'>Rodada {i + 1} de {rodadas}</div>", unsafe_allow_html=True)
            
                    # Definindo o seed para garantir reprodutibilidade
                    algorithm_globals.random_seed = 10598
            
                    # Definindo o circuito variacional de acordo com o tipo
                    if tipo_circuito == 'Real Amplitudes':
                        variational_circuit = RealAmplitudes(qubits, reps=camadas)
                    elif tipo_circuito == 'Two Local':
                        variational_circuit = TwoLocal(qubits, rotacao_escolhida, entanglement_escolhido, reps=camadas)
            
                    # Definindo o otimizador
                    if otimizador == "SPSA":
                        otimizador_instanciado = SPSA()
                    else: 
                        otimizador_instanciado = COBYLA()
            
                    # Inicializando o sampler e o solver
                    sampler = Sampler(options={"shots": shots})
                    mes = SamplingVQE(sampler=Sampler(), ansatz=variational_circuit, optimizer=otimizador_instanciado, initial_point=params)
                    meo = MinimumEigenOptimizer(min_eigen_solver=mes)
            
                    # Executando o VQE e calculando o tempo de execução
                    start = time.time()
                    vqe_result = meo.solve(qubo)
                    end = time.time()
            
                    # Armazenando os resultados
                    energias.append(vqe_result.fval)
                    tempos_execucao.append(end - start)

                # Processamento dos resultados
                energia_otimizada = min(energias)
                confiabilidade = 1 - math.exp(energia_otimizada)
                media_energia = np.mean(energias)
                desvio_padrao_energia = np.std(energias)
                tempo_medio = np.mean(tempos_execucao)
                
                indice_min_energia = energias.index(energia_otimizada)
                componente_otimo = componentes_otimos[indice_min_energia]
                
                # Decodifica os componentes escolhidos
                componentes_variaveis = []
                var_index = 0
                for m in range(1, ct + 1):
                    componente = 0
                    for k in range(nb):
                        var_value = componente_otimo[var_index]
                        componente += var_value * (2 ** (m - k - 1))
                        var_index += 1 
                    componentes_variaveis.append(componente)
                
                # Cálculo do custo total
                custo_total = sum(c * p for c, p in zip(componentes_variaveis, cjk))
                
                # Remove o GIF de carregamento
                loading_placeholder.empty()
                
                # Exibir resultados
                st.write("---")
                st.subheader("Resultados")
                st.write(f"**Energia Ótima:** {energia_otimizada}")
                st.write(f"**Confiabilidade Ótima:** {confiabilidade:.8f}")
                st.write(f"**Componentes da Solução:** {componentes_variaveis}")
                st.write(f"**Custo Total da Solução:** {custo_total}")
                
                st.subheader("Medidas Descritivas das Energias")
                st.write(f"**Média das Energias:** {media_energia}")
                st.write(f"**Desvio Padrão das Energias:** {desvio_padrao_energia}")
                st.write(f"**Tempo Médio de Execução:** {tempo_medio:.4f} segundos")
                
                # Exibir todas as energias e tempos de execução
                st.subheader("Energias e Tempos por Execução")
                for i in range(rodadas):
                    st.write(f"Rodada {i+1}: Energia = {energias[i]}, Tempo = {tempos_execucao[i]:.4f} segundos")
                
                # Exibir o circuito do VQE
                st.subheader("Circuito do VQE")
                st.write(mes.ansatz.decompose().draw(output='mpl'))

            # Botão de Reset
        if st.button('Reset'):
            # Limpar variáveis ou realizar qualquer ação de reset necessária
            st.experimental_rerun()
    
if __name__ == '__main__':
    main()
