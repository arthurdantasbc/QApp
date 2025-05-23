import streamlit as st

# Dicionário simples com textos em PT e EN
TEXTOS = {
    "pt": {
        "titulo": "qxplore",
        "escolha_area": "Escolha uma área de aplicação:",
        "otimizacao": "Otimização",
        "monitoramento": "Monitoramento",
        "manutencao": "Manutenção",
        "pagina_otimizacao": "Página de Otimização",
        "pagina_monitoramento": "Página de Monitoramento",
        "pagina_manutencao": "Página de Manutenção",
        "instancia_input": "Digite alguma coisa para testar a instância:",
        "instancia_recebida": "Instância recebida:",
        "ajuda": "Use os botões para navegar entre as áreas de aplicação.",
        "idioma": "Escolha o idioma:"
    },
    "en": {
        "titulo": "qxplore",
        "escolha_area": "Choose an application area:",
        "otimizacao": "Optimization",
        "monitoramento": "Monitoring",
        "manutencao": "Maintenance",
        "pagina_otimizacao": "Optimization Page",
        "pagina_monitoramento": "Monitoring Page",
        "pagina_manutencao": "Maintenance Page",
        "instancia_input": "Type something to test the instance:",
        "instancia_recebida": "Received instance:",
        "ajuda": "Use the buttons to navigate between application areas.",
        "idioma": "Choose the language:"
    }
}

def mostrar_ajuda(textos):
    st.sidebar.info(textos["ajuda"])

def mostrar_cartoes_de_area(textos):
    st.title(textos["titulo"])
    
    st.subheader(textos["escolha_area"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("opt.png", use_container_width=True)
        if st.button(textos["otimizacao"], key="otimizacao_btn"):
            st.session_state['pagina'] = 'otimizacao'

    with col2:
        st.image("ml.png", use_container_width=True)
        if st.button(textos["monitoramento"], key="monitoramento_btn"):
            st.session_state['pagina'] = 'monitoramento'

    with col3:
        st.image("infer.png", use_container_width=True)
        if st.button(textos["manutencao"], key="manutencao_btn"):
            st.session_state['pagina'] = 'manutencao'

def ler_manualmente(textos):
    valor = st.text_input(textos["instancia_input"])
    if valor:
        return {"valor": valor}
    return None

def mostrar_instancia(instancia, textos):
    st.write(textos["instancia_recebida"])
    st.json(instancia)

def main():
    # Seleção do idioma no sidebar (antes de qualquer coisa)
    idioma = st.sidebar.selectbox("🌐 " + TEXTOS["pt"]["idioma"], ("Português", "English"))
    if idioma == "Português":
        lang = "pt"
    else:
        lang = "en"

    textos = TEXTOS[lang]

    mostrar_ajuda(textos)

    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = 'inicio'

    if st.session_state['pagina'] == 'inicio':
        mostrar_cartoes_de_area(textos)

    elif st.session_state['pagina'] == 'otimizacao':
        st.subheader(textos["pagina_otimizacao"])
        instancia = ler_manualmente(textos)
        if instancia:
            mostrar_instancia(instancia, textos)

    elif st.session_state['pagina'] == 'monitoramento':
        st.subheader(textos["pagina_monitoramento"])
        st.write("Conteúdo da página de monitoramento aqui.")

    elif st.session_state['pagina'] == 'manutencao':
        st.subheader(textos["pagina_manutencao"])
        st.write("Conteúdo da página de manutenção aqui.")

if __name__ == "__main__":
    main()
