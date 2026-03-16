import streamlit as st
from utils.auth import check_password

st.set_page_config(
    page_title="Ma Toolbox SEO",
    page_icon="🧰",
    layout="wide"
)

check_password()

st.title("🧰 Ma Toolbox SEO")
st.markdown("*Tous mes outils SEO regroupés en une seule application.*")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### ✍️ Rédaction Contenu
    **SEO Content Writer Pro V2**

    Outil complet de rédaction SEO :
    - Analyse concurrentielle SERP via DataForSEO
    - Génération de structure Hn optimisée
    - Rédaction paragraphe par paragraphe avec IA
    - Gestion du maillage interne via sitemap
    - Export en .docx

    👉 Accède via la **sidebar** à gauche
    """)

with col2:
    st.markdown("""
    ### 🔍 Audit SEO Technique
    **SEO Technical Audit Tool Expert**

    Audit technique avancé :
    - Import croisé GSC + Screaming Frog
    - Analyse HTTP, balises, contenu, maillage
    - Détection pages zombies, quick wins
    - Analyse 404 et redirections GSC
    - Rapport IA détaillé via Claude

    👉 Accède via la **sidebar** à gauche
    """)

with col3:
    st.markdown("""
    ### 🎯 Analyse Concurrentielle
    **Détection de concurrents organiques**

    Identifie tes vrais rivaux SEO :
    - Détection automatique via DataForSEO Labs
    - Score de pertinence multi-critères
    - Analyse des mots-clés partagés
    - Graphiques radar comparatifs
    - Export CSV

    👉 Accède via la **sidebar** à gauche
    """)

st.divider()

st.markdown("""
### ⚙️ Configuration requise

Pour utiliser ces outils, tu auras besoin de configurer les **secrets** suivants
dans les paramètres de l'app (Settings > Secrets sur Streamlit Community Cloud) :

```toml
DATAFORSEO_USERNAME = "ton_username"
DATAFORSEO_PASSWORD = "ton_password"
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "AI..."
```

Les clés sont pré-remplies automatiquement dans chaque outil si elles sont configurées dans les secrets.
Tu peux aussi les saisir manuellement dans la sidebar de chaque outil.
""")

st.caption("🧰 Ma Toolbox SEO | Propulsé par Streamlit")
