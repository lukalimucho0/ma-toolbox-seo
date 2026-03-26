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

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✍️ Rédaction Contenu
    **SEO Content Writer Pro V2**

    - Analyse SERP via DataForSEO
    - Structure Hn optimisée + rédaction IA
    - Maillage interne via sitemap
    - Export .docx
    """)

with col2:
    st.markdown("""
    ### 🔍 Audit SEO Technique
    **SEO Technical Audit Tool Expert**

    - Import croisé GSC + Screaming Frog
    - Détection pages zombies, quick wins
    - Analyse 404 et redirections GSC
    - Rapport IA détaillé via Claude
    """)

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    ### 🎯 Analyse Concurrentielle
    **Concurrents SEO + Business**

    - DataForSEO Labs + Claude IA
    - Score composite 40% SEO + 60% Business
    - Détection concurrents hors radar SEO
    - Graphiques radar + Export CSV
    """)

with col4:
    st.markdown("""
    ### 🔄 Rafraîchissement Contenu
    **Outil dédié Gérer Seul**

    - Scraping article existant + concurrents
    - Structure optimale par Claude
    - Rédaction rafraîchie avec maillage interne
    - Mode bulk (5 articles) + Export .docx
    """)

st.markdown("*👉 Accède aux outils via la **sidebar** à gauche*")

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
