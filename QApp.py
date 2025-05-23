def mostrar_cartoes_de_area():
    st.subheader("Escolha uma área de aplicação:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("img/otimizacao.png", use_container_width=True)
        if st.button("Otimização", key="otimizacao_btn"):
            st.session_state['pagina'] = 'otimizacao'

    with col2:
        st.image("img/monitoramento.png", use_container_width=True)
        if st.button("Monitoramento", key="monitoramento_btn"):
            st.session_state['pagina'] = 'monitoramento'

    with col3:
        st.image("img/manutencao.png", use_container_width=True)
        if st.button("Manutenção", key="manutencao_btn"):
            st.session_state['pagina'] = 'manutencao'
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
        # Conteúdo correspondente

    elif st.session_state['pagina'] == 'manutencao':
        st.subheader("Página de Manutenção")
        # Conteúdo correspondente
