"""
🎯 ANALYSE CONCURRENTIELLE SEO
================================
Détection automatique des meilleurs concurrents organiques d'un domaine
via l'API DataForSEO Labs.
"""

import streamlit as st
import requests
import base64
import pandas as pd
import plotly.graph_objects as go
from urllib.parse import urlparse
import re

from utils.auth import check_password

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Analyse Concurrentielle | Ma Toolbox SEO",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

check_password()

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    .competitor-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 4px solid #667eea;
    }
    .rank-badge {
        font-size: 2em;
        font-weight: bold;
    }
    .metric-box {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CLASSE API
# =============================================================================
class DataForSEOLabs:
    """Client pour l'API DataForSEO Labs - Competitors Domain."""

    def __init__(self, username: str, password: str):
        self.base_url = "https://api.dataforseo.com/v3"
        self.session = requests.Session()
        credentials = f"{username}:{password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.session.headers.update({
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json"
        })

    def get_competitors(self, domain: str, location_code: int, language_code: str,
                        limit: int = 30, exclude_top_domains: bool = True) -> dict:
        """Récupère les concurrents organiques d'un domaine."""
        endpoint = f"{self.base_url}/dataforseo_labs/google/competitors_domain/live"
        payload = [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "exclude_top_domains": exclude_top_domains,
            "item_types": ["organic"],
            "filters": [
                "metrics.organic.count", ">", 5
            ],
            "order_by": ["metrics.organic.count,desc"]
        }]
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()

    def get_domain_metrics(self, domain: str, location_code: int, language_code: str) -> dict:
        """Récupère les métriques globales d'un domaine (pour le site analysé)."""
        endpoint = f"{self.base_url}/dataforseo_labs/google/competitors_domain/live"
        payload = [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": 1,
            "item_types": ["organic"]
        }]
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================
def clean_domain(url: str) -> str:
    """Nettoie une URL pour n'en garder que le domaine."""
    url = url.strip().lower()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    domain = re.sub(r'^www\.', '', domain)
    domain = domain.rstrip('/')
    return domain


def calculate_relevance_score(competitor: dict, target_keywords_count: int) -> float:
    """
    Calcule un score de pertinence (0-100) pour un concurrent.

    Le score combine :
    - Taux d'intersection des mots-clés (50%) : plus le concurrent partage de mots-clés
      avec le site analysé, plus il est pertinent comme concurrent direct.
    - Position moyenne sur les mots-clés partagés (25%) : un concurrent qui se positionne
      bien sur les mêmes mots-clés est un rival plus direct.
    - Proportion de trafic issu de l'intersection (25%) : si une grande part du trafic
      du concurrent vient des mots-clés partagés, leurs stratégies sont similaires.
    """
    intersections = competitor.get("intersections", 0)
    avg_position = competitor.get("avg_position", 50)

    metrics_organic = competitor.get("metrics", {}).get("organic", {})
    full_metrics_organic = competitor.get("full_domain_metrics", {}).get("organic", {})

    intersection_count = metrics_organic.get("count", 0)
    full_count = full_metrics_organic.get("count", 1)
    intersection_etv = metrics_organic.get("etv", 0)
    full_etv = full_metrics_organic.get("etv", 1)

    # 1. Taux d'intersection par rapport aux mots-clés du site analysé
    if target_keywords_count > 0:
        intersection_ratio = min(intersections / target_keywords_count, 1.0)
    else:
        intersection_ratio = min(intersection_count / max(full_count, 1), 1.0)

    # 2. Score de position (meilleure position = meilleur score)
    position_score = max(0, (50 - avg_position) / 50) if avg_position <= 50 else 0

    # 3. Ratio de trafic partagé (quelle part du trafic du concurrent vient de l'intersection)
    traffic_overlap_ratio = min(intersection_etv / max(full_etv, 1), 1.0)

    score = (intersection_ratio * 50) + (position_score * 25) + (traffic_overlap_ratio * 25)
    return round(min(score, 100), 1)


def format_number(n: float) -> str:
    """Formate un nombre pour l'affichage."""
    if n is None:
        return "N/A"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


LOCATIONS = {
    "France": {"code": 2250, "lang": "fr"},
    "Belgique": {"code": 2056, "lang": "fr"},
    "Suisse": {"code": 2756, "lang": "fr"},
    "Canada (FR)": {"code": 2124, "lang": "fr"},
    "États-Unis": {"code": 2840, "lang": "en"},
    "Royaume-Uni": {"code": 2826, "lang": "en"},
    "Allemagne": {"code": 2276, "lang": "de"},
    "Espagne": {"code": 2724, "lang": "es"},
    "Italie": {"code": 2380, "lang": "it"},
    "Portugal": {"code": 2620, "lang": "pt"},
    "Pays-Bas": {"code": 2528, "lang": "nl"},
}


# =============================================================================
# INTERFACE
# =============================================================================
st.title("🎯 Analyse Concurrentielle SEO")
st.markdown("*Détecte automatiquement les 5 meilleurs concurrents organiques d'un site.*")

# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Configuration")

    dataforseo_username = st.text_input(
        "Username DataForSEO",
        value=st.secrets.get("DATAFORSEO_USERNAME", ""),
        type="password"
    )
    dataforseo_password = st.text_input(
        "Password DataForSEO",
        value=st.secrets.get("DATAFORSEO_PASSWORD", ""),
        type="password"
    )

    st.divider()
    st.header("🎯 Paramètres d'analyse")

    target_url = st.text_input(
        "Domaine à analyser",
        placeholder="exemple.fr",
        help="Entre le domaine sans https:// ni www."
    )

    selected_location = st.selectbox("Pays cible", list(LOCATIONS.keys()), index=0)

    exclude_top = st.checkbox(
        "Exclure les gros portails",
        value=True,
        help="Exclut Wikipedia, Amazon, YouTube, etc."
    )

    nb_competitors = st.slider("Nombre de concurrents à afficher", 3, 10, 5)

    st.divider()
    st.markdown("""
    ### Comment ça marche ?
    1. On récupère tous les domaines qui se positionnent sur les mêmes mots-clés que toi
    2. On calcule un **score de pertinence** basé sur :
       - Le taux d'intersection des mots-clés
       - La position moyenne sur ces mots-clés
       - La proportion de trafic partagé
    3. On classe et on ne garde que le **top {}**
    """.format(nb_competitors))

# ─── Zone principale ───
if not dataforseo_username or not dataforseo_password:
    st.warning("Configure tes identifiants DataForSEO dans la sidebar.")
    st.stop()

if not target_url:
    st.info("Entre un domaine dans la sidebar pour lancer l'analyse.")
    st.stop()

domain = clean_domain(target_url)
st.markdown(f"**Domaine analysé** : `{domain}`")

if st.button("🚀 Lancer l'analyse concurrentielle", type="primary", use_container_width=True):
    api = DataForSEOLabs(dataforseo_username, dataforseo_password)
    loc = LOCATIONS[selected_location]

    with st.status("Analyse en cours...", expanded=True) as status:
        # ── Étape 1 : Récupération des concurrents ──
        st.write("🔍 Recherche des concurrents organiques...")
        try:
            result = api.get_competitors(
                domain=domain,
                location_code=loc["code"],
                language_code=loc["lang"],
                limit=50,
                exclude_top_domains=exclude_top
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                st.error("Identifiants DataForSEO invalides.")
            elif e.response.status_code == 403:
                st.error("Accès refusé — vérifie tes crédits DataForSEO.")
            else:
                st.error(f"Erreur API : {e}")
            st.stop()
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.stop()

        # Vérification de la réponse
        tasks = result.get("tasks", [])
        if not tasks or tasks[0].get("status_code") != 20000:
            error_msg = tasks[0].get("status_message", "Erreur inconnue") if tasks else "Pas de réponse"
            st.error(f"Erreur DataForSEO : {error_msg}")
            st.stop()

        task_result = tasks[0].get("result", [])
        if not task_result:
            st.warning("Aucun résultat trouvé pour ce domaine. Vérifie que le domaine est correct et a suffisamment de visibilité organique.")
            st.stop()

        items = task_result[0].get("items", [])
        total_count = task_result[0].get("total_count", 0)

        if not items:
            st.warning("Aucun concurrent trouvé. Le domaine n'a peut-être pas assez de mots-clés indexés.")
            st.stop()

        st.write(f"✅ {total_count} concurrents potentiels trouvés, analyse des meilleurs...")

        # ── Étape 2 : Calcul des scores ──
        st.write("📊 Calcul des scores de pertinence...")

        # Extraire le nombre de mots-clés du site cible depuis les résultats
        # (on utilise le se_type info ou le total_count comme proxy)
        target_keywords_count = task_result[0].get("total_count", 100)

        competitors = []
        for item in items:
            comp_domain = item.get("domain", "")
            if comp_domain == domain:
                continue

            score = calculate_relevance_score(item, target_keywords_count)
            metrics_organic = item.get("metrics", {}).get("organic", {})
            full_metrics = item.get("full_domain_metrics", {}).get("organic", {})

            competitors.append({
                "domain": comp_domain,
                "score": score,
                "intersections": item.get("intersections", 0),
                "avg_position": round(item.get("avg_position", 0), 1),
                "shared_keywords": metrics_organic.get("count", 0),
                "shared_etv": metrics_organic.get("etv", 0),
                "total_keywords": full_metrics.get("count", 0),
                "total_etv": full_metrics.get("etv", 0),
                "estimated_paid_cost": full_metrics.get("estimated_paid_traffic_cost", 0),
                "pos_1": metrics_organic.get("pos_1", 0),
                "pos_2_3": metrics_organic.get("pos_2_3", 0),
                "pos_4_10": metrics_organic.get("pos_4_10", 0),
                "raw": item
            })

        # Trier par score de pertinence
        competitors.sort(key=lambda x: x["score"], reverse=True)
        top_competitors = competitors[:nb_competitors]

        status.update(label="Analyse terminée !", state="complete", expanded=False)

    # =====================================================================
    # AFFICHAGE DES RÉSULTATS
    # =====================================================================
    st.divider()
    st.header(f"🏆 Top {nb_competitors} concurrents organiques de `{domain}`")
    st.caption(f"Pays : {selected_location} — {total_count} concurrents analysés")

    # ── Vue d'ensemble ──
    cols = st.columns(nb_competitors)
    for i, comp in enumerate(top_competitors):
        with cols[i]:
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            medal = medals[i] if i < len(medals) else f"#{i+1}"
            st.metric(
                label=f"{medal} {comp['domain']}",
                value=f"{comp['score']}/100",
                delta=f"{comp['intersections']} mots-clés communs"
            )

    st.divider()

    # ── Détail par concurrent ──
    for i, comp in enumerate(top_competitors):
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        medal = medals[i] if i < len(medals) else f"#{i+1}"

        with st.expander(f"{medal} **{comp['domain']}** — Score : {comp['score']}/100", expanded=(i < 3)):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Mots-clés communs", format_number(comp["intersections"]))
            c2.metric("Position moy.", f"{comp['avg_position']}")
            c3.metric("Trafic partagé", format_number(comp["shared_etv"]))
            c4.metric("Trafic total", format_number(comp["total_etv"]))
            c5.metric("Mots-clés totaux", format_number(comp["total_keywords"]))

            # Distribution des positions sur les mots-clés partagés
            pos_data = {
                "Top 1": comp["pos_1"],
                "Top 2-3": comp["pos_2_3"],
                "Top 4-10": comp["pos_4_10"],
            }

            col_chart, col_info = st.columns([1, 1])

            with col_chart:
                fig = go.Figure(data=[go.Bar(
                    x=list(pos_data.keys()),
                    y=list(pos_data.values()),
                    marker_color=["#667eea", "#764ba2", "#f093fb"]
                )])
                fig.update_layout(
                    title="Positions sur les mots-clés partagés",
                    height=280,
                    margin=dict(t=40, b=20, l=20, r=20),
                    yaxis_title="Nombre de mots-clés"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_info:
                overlap_pct = round(comp["intersections"] / max(comp["total_keywords"], 1) * 100, 1)
                traffic_shared_pct = round(comp["shared_etv"] / max(comp["total_etv"], 1) * 100, 1)

                st.markdown(f"""
                **Analyse rapide :**
                - **{comp['intersections']}** mots-clés en commun avec `{domain}`
                - **{overlap_pct}%** de ses mots-clés chevauchent les tiens
                - **{traffic_shared_pct}%** de son trafic vient de ces mots-clés partagés
                - Valeur publicitaire estimée du domaine : **{format_number(comp['estimated_paid_cost'])}€/mois**
                """)

                if comp["avg_position"] < 10:
                    st.warning("Concurrent très bien positionné — rival direct")
                elif comp["avg_position"] < 20:
                    st.info("Concurrent bien positionné — à surveiller")

    # ── Tableau comparatif ──
    st.divider()
    st.subheader("📋 Tableau comparatif")

    df = pd.DataFrame([{
        "Rang": i + 1,
        "Domaine": c["domain"],
        "Score": c["score"],
        "Mots-clés communs": c["intersections"],
        "Position moy.": c["avg_position"],
        "Trafic partagé (est.)": round(c["shared_etv"]),
        "Trafic total (est.)": round(c["total_etv"]),
        "Mots-clés totaux": c["total_keywords"],
    } for i, c in enumerate(top_competitors)])

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.1f"
            ),
        }
    )

    # ── Graphique radar ──
    st.subheader("🕸️ Comparaison radar")

    # Normalisation pour le radar
    max_intersections = max(c["intersections"] for c in top_competitors) or 1
    max_etv = max(c["shared_etv"] for c in top_competitors) or 1
    max_total_kw = max(c["total_keywords"] for c in top_competitors) or 1

    fig_radar = go.Figure()
    colors = ["#667eea", "#f093fb", "#4fd1c5", "#f6ad55", "#fc8181",
              "#68d391", "#b794f4", "#63b3ed", "#fbb6ce", "#c6f6d5"]

    for i, comp in enumerate(top_competitors):
        position_score = max(0, (50 - comp["avg_position"]) / 50 * 100)

        fig_radar.add_trace(go.Scatterpolar(
            r=[
                comp["intersections"] / max_intersections * 100,
                position_score,
                comp["shared_etv"] / max_etv * 100,
                comp["total_keywords"] / max_total_kw * 100,
                comp["score"],
            ],
            theta=["Mots-clés communs", "Position", "Trafic partagé",
                   "Mots-clés totaux", "Score global"],
            fill="toself",
            name=comp["domain"],
            line_color=colors[i % len(colors)],
            opacity=0.6
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=500,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Export CSV ──
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter en CSV",
        data=csv,
        file_name=f"concurrents_{domain}_{selected_location}.csv",
        mime="text/csv"
    )

    # ── Coût API ──
    cost = result.get("cost", 0)
    st.caption(f"Coût API DataForSEO : ${cost:.4f}")

st.caption("🎯 Analyse Concurrentielle SEO | Ma Toolbox SEO")
