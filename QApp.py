import streamlit as st

# Textos multilíngues
TEXTOS = {
    "pt": {
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
    st.subheader(textos["escolha_area"])

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
    col1, col2, col3, col3, col5 = st.columns([1, 2,2, 2, 1])
    with col3:
        st.image("qxplore.png", use_container_width=True)

def main():
    st.set_page_config(page_title="qxplore", layout="wide")

    idioma = st.sidebar.selectbox("🌐 " + TEXTOS["pt"]["idioma"], ("Português", "English"))
    lang = "pt" if idioma == "Português" else "en"
    textos = TEXTOS[lang]

    mostrar_ajuda(textos)
    mostrar_logo_topo()

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
        st.write("Conteúdo da página de monitoramento.")

    elif st.session_state['pagina'] == 'manutencao':
        st.subheader(textos["pagina_manutencao"])
        st.write("Conteúdo da página de manutenção.")

if __name__ == "__main__":
    main()
