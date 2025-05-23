import streamlit as st

def mostrar_ajuda():
    st.sidebar.info("Use os botões para navegar entre as áreas de aplicação.")

def mostrar_cartoes_de_area():
    st.subheader("Escolha uma área de aplicação:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://via.placeholder.com/150?text=Otimizacao", use_container_width=True)
        if st.button("Otimização", key="otimizacao_btn"):
            st.session_state['pagina'] = 'otimizacao'

    with col2:
        st.image("https://via.placeholder.com/150?text=Monitoramento", use_container_width=True)
        if st.button("Monitoramento", key="monitoramento_btn"):
            st.session_state['pagina'] = 'monitoramento'

    with col3:
        st.image("https://via.placeholder.com/150?text=Manutencao", use_container_width=True)
        if st.button("Manutenção", key="manutencao_btn"):
            st.session_state['pagina'] = 'manutencao'

def ler_manualmente():
    # Apenas um exemplo simples para testar
    valor = st.text_input("Digite alguma coisa para testar a instância:")
    if valor:
        return {"valor": valor}
    return None

def mostrar_instancia(instancia):
    st.write("Instância recebida:")
    st.json(instancia)

def main():
    mostrar_ajuda()

    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = 'inicio'

    if st.session_state['pagina'] == 'inicio':
        mostrar_cartoes_de_area()

    elif st.session_state['pagina'] == 'otimizacao':
        st.subheader("Página de Otimização")
        instancia = ler_manualmente()
        if instancia:
            mostrar_instancia(instancia)

    elif st.session_state['pagina'] == 'monitoramento':
        st.subheader("Página de Monitoramento")
        st.write("Conteúdo da página de monitoramento aqui.")

    elif st.session_state['pagina'] == 'manutencao':
        st.subheader("Página de Manutenção")
        st.write("Conteúdo da página de manutenção aqui.")

if __name__ == "__main__":
    main()
