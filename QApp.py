import streamlit as st

# Textos multilíngues
TEXTOS = {
    "pt": {
        
        "pagina_otimizacao": "Página de Otimização",
        "pagina_monitoramento": "Página de Monitoramento",
        "pagina_manutencao": "Página de Manutenção",
        "instancia_input": "Digite alguma coisa para testar a instância:",
        "instancia_recebida": "Instância recebida:",
        "ajuda": "Clique em uma imagem para navegar.",
        "idioma": "Escolha o idioma:"
    },
    "en": {
        
        "pagina_otimizacao": "Optimization Page",
        "pagina_monitoramento": "Monitoring Page",
        "pagina_manutencao": "Maintenance Page",
        "instancia_input": "Type something to test the instance:",
        "instancia_recebida": "Received instance:",
        "ajuda": "Click an image to navigate.",
        "idioma": "Choose the language:"
    }
}

def mostrar_introducao_e_titulo(textos):
    texto = textos['intro']
    st.markdown(
        f"""
        <div style="text-align: center; font-size:16px; color: gray; white-space: pre-line; line-height:1.5;">
            <br>
            {texto}
        </div>
        """,
        unsafe_allow_html=True
    )
    
def mostrar_referencias():
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

def mostrar_cartoes_de_area(textos):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("")
    with col2:
        if st.button("", key="otimizacao_btn"):
            st.session_state['pagina'] = 'otimizacao'
        st.image("opt.png", width=200)
    with col3:
        if st.button("", key="monitoramento_btn"):
            st.session_state['pagina'] = 'monitoramento'
        st.image("ml.png", width=200)
    with col4:
        if st.button("", key="manutencao_btn"):
            st.session_state['pagina'] = 'manutencao'
        st.image("infer.png", width=200)
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

def main():
    st.set_page_config(page_title="qxplore", layout="wide")

    # 1 - imagem no topo da sidebar
    st.sidebar.image("CM.png", use_container_width=True)

    # 2 - escolha de idioma logo abaixo da imagem
    idioma = st.sidebar.selectbox("🌐 " + TEXTOS["pt"]["idioma"], ("Português", "English"))
    lang = "pt" if idioma == "Português" else "en"
    textos = TEXTOS[lang]

    # 3 - aviso para clicar na imagem
    st.sidebar.info(textos["ajuda"])

    # 4 - referências em expander
    mostrar_referencias()

    mostrar_logo_topo()

    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = 'inicio'

    if st.session_state['pagina'] == 'inicio':
        mostrar_introducao_e_titulo(textos)
        mostrar_cartoes_de_area(textos)

    elif st.session_state['pagina'] == 'otimizacao':
        st.subheader(textos["pagina_otimizacao"])
        instancia = ler_manualmente(textos)
        if instancia:
            mostrar_instancia(instancia, textos)

    elif st.session_state['pagina'] == 'monitoramento':
        st.subheader(textos["pagina_monitoramento"])
        st.write("Conteúdo da página de monitoramento.")

    elif st.session_state['pagina'] == 'manutencao':
        st.subheader(textos["pagina_manutencao"])
        st.write("Conteúdo da página de manutenção.")

if __name__ == "__main__":
    main()
