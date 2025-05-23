import streamlit as st

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

def mostrar_introducao_e_titulo(textos):
    texto = textos['intro']
    st.markdown(
        f"""
        <div style="text-align: center; font-size:16px; color: gray; line-height:1.5;white-space: pre-line;">
            {texto}
        </div>
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
        if st.button(textos["pagina_otimizacao"], key="otimizacao_btn"):
            st.session_state['pagina'] = 'otimizacao'
        st.image("opt.png", width=200)
    with col3:
        if st.button(textos["pagina_ml"], key="ml_btn"):
            st.session_state['pagina'] = 'ml'
        st.image("ml.png", width=200)
    with col4:
        if st.button(textos["pagina_inferencia"], key="inferencia_btn"):
            st.session_state['pagina'] = 'inferencia'
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
    mostrar_referencias(textos)

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

    elif st.session_state['pagina'] == 'ml':
        st.subheader(textos["pagina_ml"])
        st.write("Conteúdo da página de Machine Learning Quântico.")

    elif st.session_state['pagina'] == 'inferencia':
        st.subheader(textos["pagina_inferencia"])
        st.write("Conteúdo da página de Inferência Quântica.")

if __name__ == "__main__":
    main()
