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
    # Título é uma imagem clicável que não muda de página, só mostra o título visual
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="qxplore.png" alt="qxplore" style="width:250px;"/>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.subheader(textos["escolha_area"])

    col1, col2, col3 = st.columns(3)

    # Função para criar imagens clicáveis que mudam a página
    def imagem_clicavel(img_src, alt_text, pagina):
        codigo_html = f'''
        <div style="text-align:center; cursor:pointer;">
            <img src="{img_src}" alt="{alt_text}" style="width:150px;" 
                onclick="window.parent.location.href='#{pagina}'"/>
        </div>
        '''
        st.markdown(codigo_html, unsafe_allow_html=True)
    
    # No Streamlit o clique na imagem não funciona de verdade, então vamos usar botões transparentes com imagens
    # Melhor solução: colocar a imagem e um botão abaixo transparente para clicar, ou o botão com imagem dentro

    with col1:
        if st.button(textos["otimizacao"], key="btn_otimizacao", help=textos["otimizacao"]):
            st.session_state['pagina'] = 'otimizacao'
        st.image("opt.png", use_container_width=True)

    with col2:
        if st.button(textos["monitoramento"], key="btn_monitoramento", help=textos["monitoramento"]):
            st.session_state['pagina'] = 'monitoramento'
        st.image("ml.png", use_container_width=True)

    with col3:
        if st.button(textos["manutencao"], key="btn_manutencao", help=textos["manutencao"]):
            st.session_state['pagina'] = 'manutencao'
        st.image("infer.png", use_container_width=True)

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

    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = 'inicio'

    # Se tiver hash na URL, usa para mudar a página (simula clique na imagem)
    # Isso ajuda clicar na imagem com o código html acima
    pagina_hash = st.experimental_get_query_params().get("pagina", [None])[0]
    if pagina_hash:
        st.session_state['pagina'] = pagina_hash

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
