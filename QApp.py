import streamlit as st

# Textos multilíngues
TEXTOS = {
    "pt": {
        "intro": "Seja bem vindo ao QXplore, este aplicativo foi desenvolvido com o intuito de promover a utilização da computação quântica em 3 áreas diferentes, conforme descrito abaixo. Selecione a área que deseja explorar.",
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
        "intro": "Welcome to QXplore! This application was developed to promote the use of quantum computing in 3 different areas, described below. Select the area you want to explore.",
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
def mostrar_introducao_e_titulo(textos):
    st.markdown(
        f"<p style='font-size:16px; color: gray;'>{textos['intro']}</p>", 
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h3 style='font-weight: 600; margin-top: 14px;'>{textos['escolha_area']}</h3>", 
        unsafe_allow_html=True
    )
def mostrar_ajuda(textos):
    st.sidebar.info(textos["ajuda"])

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
    col1, col2, col3 = st.columns([1, 2, 1])  # col2 é maior, fica no centro
    with col2:
        st.image("qxplore.png", width=600)

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
