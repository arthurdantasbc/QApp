import streamlit as st

# Textos multilíngues
TEXTOS = {
    "pt": {
        "titulo": "qxplore",
        "escolha_area": "Escolha uma área de aplicação:",
        "pagina_otimizacao": "Página de Otimização",
        "pagina_monitoramento": "Página de Monitoramento",
        "pagina_manutencao": "Página de Manutenção",
        "instancia_input": "Digite alguma coisa para testar a instância:",
        "instancia_recebida": "Instância recebida:",
        "ajuda": "Clique em uma imagem para navegar.",
        "idioma": "Escolha o idioma:"
    },
    "en": {
        "titulo": "qxplore",
        "escolha_area": "Choose an application area:",
        "pagina_otimizacao": "Optimization Page",
        "pagina_monitoramento": "Monitoring Page",
        "pagina_manutencao": "Maintenance Page",
        "instancia_input": "Type something to test the instance:",
        "instancia_recebida": "Received instance:",
        "ajuda": "Click an image to navigate.",
        "idioma": "Choose the language:"
    }
}

def mostrar_ajuda(textos):
    st.sidebar.info(textos["ajuda"])

def mostrar_cartoes_de_area(textos):
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="qxplore.png" alt="qxplore" style="width:200px;"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(textos["escolha_area"])

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.markdown(
            f"""
            <a href="?pagina=otimizacao">
                <img src="opt.png" alt="Otimização" style="width:100%; border-radius:10px;"/>
            </a>
            """, unsafe_allow_html=True):
            pass

    with col2:
        if st.markdown(
            f"""
            <a href="?pagina=monitoramento">
                <img src="ml.png" alt="Monitoramento" style="width:100%; border-radius:10px;"/>
            </a>
            """, unsafe_allow_html=True):
            pass

    with col3:
        if st.markdown(
            f"""
            <a href="?pagina=manutencao">
                <img src="infer.png" alt="Manutenção" style="width:100%; border-radius:10px;"/>
            </a>
            """, unsafe_allow_html=True):
            pass

def ler_manualmente(textos):
    valor = st.text_input(textos["instancia_input"])
    if valor:
        return {"valor": valor}
    return None

def mostrar_instancia(instancia, textos):
    st.write(textos["instancia_recebida"])
    st.json(instancia)

def main():
    idioma = st.sidebar.selectbox("🌐 " + TEXTOS["pt"]["idioma"], ("Português", "English"))
    lang = "pt" if idioma == "Português" else "en"
    textos = TEXTOS[lang]

    mostrar_ajuda(textos)

    # Define página padrão
    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = 'inicio'

    # Detecta troca de página via URL (?pagina=otimizacao, etc)
    pagina_hash = st.query_params.get("pagina", [None])[0]
    if pagina_hash:
        st.session_state['pagina'] = pagina_hash

    # Roteamento
    if st.session_state['pagina'] == 'inicio':
        mostrar_cartoes_de_area(textos)

    elif st.session_state['pagina'] == 'otimizacao':
        st.markdown(
            '<div style="text-align:center;"><img src="qxplore.png" style="width:150px;"/></div>',
            unsafe_allow_html=True
        )
        st.subheader(textos["pagina_otimizacao"])
        instancia = ler_manualmente(textos)
        if instancia:
            mostrar_instancia(instancia, textos)

    elif st.session_state['pagina'] == 'monitoramento':
        st.markdown(
            '<div style="text-align:center;"><img src="qxplore.png" style="width:150px;"/></div>',
            unsafe_allow_html=True
        )
        st.subheader(textos["pagina_monitoramento"])
        st.write("Conteúdo da página de monitoramento.")

    elif st.session_state['pagina'] == 'manutencao':
        st.markdown(
            '<div style="text-align:center;"><img src="qxplore.png" style="width:150px;"/></div>',
            unsafe_allow_html=True
        )
        st.subheader(textos["pagina_manutencao"])
        st.write("Conteúdo da página de manutenção.")

if __name__ == "__main__":
    main()
