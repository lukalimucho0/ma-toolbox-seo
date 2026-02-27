import streamlit as st


def check_password():
    """Vérifie le mot de passe. Retourne True si authentifié, False sinon."""

    if st.session_state.get("authenticated"):
        return True

    st.title("🔒 Accès protégé")
    st.markdown("Entre le mot de passe pour accéder à la toolbox.")

    password = st.text_input("Mot de passe", type="password", key="password_input")

    if st.button("Se connecter", type="primary"):
        if password == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")

    st.stop()
