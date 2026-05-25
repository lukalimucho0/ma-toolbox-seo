"""
🗺️ TOPICAL AUTHORITY MAP
=========================
Cartographie l'autorité thématique d'un domaine : clusters sémantiques,
couverture vs concurrents, gaps de contenu, priorisation stratégique.
Combine DataForSEO Labs (keywords) + Claude (clustering IA) + Plotly (visualisation).
"""

import streamlit as st
import requests
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from urllib.parse import urlparse
import anthropic
import json
import re
import math
from collections import defaultdict

from utils.auth import check_password

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Topical Authority Map | Ma Toolbox SEO",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

check_password()

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    .stMetric > div { padding: 8px; }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A5F;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .cluster-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .gap-card {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .opportunity-card {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CLASSE API DATAFORSEO
# =============================================================================
class DataForSEOLabs:
    """Client pour l'API DataForSEO Labs."""

    def __init__(self, username: str, password: str):
        self.base_url = "https://api.dataforseo.com/v3"
        self.session = requests.Session()
        credentials = f"{username}:{password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.session.headers.update({
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json"
        })
        self.total_cost = 0.0

    def _post(self, endpoint: str, payload: list) -> dict:
        response = self.session.post(f"{self.base_url}{endpoint}", json=payload)
        response.raise_for_status()
        data = response.json()
        self.total_cost += data.get("cost", 0)
        return data

    def get_ranked_keywords(self, domain: str, location_code: int, language_code: str,
                            limit: int = 200) -> dict:
        return self._post("/dataforseo_labs/google/ranked_keywords/live", [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "item_types": ["organic"],
            "order_by": ["keyword_data.keyword_info.search_volume,desc"]
        }])

    def get_competitors(self, domain: str, location_code: int, language_code: str,
                        limit: int = 10, exclude_top_domains: bool = True) -> dict:
        return self._post("/dataforseo_labs/google/competitors_domain/live", [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "exclude_top_domains": exclude_top_domains,
            "item_types": ["organic"],
            "filters": ["metrics.organic.count", ">", 5],
            "order_by": ["metrics.organic.count,desc"]
        }])

    def get_keywords_for_domain(self, domain: str, location_code: int, language_code: str,
                                limit: int = 200) -> dict:
        return self._post("/dataforseo_labs/google/ranked_keywords/live", [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "item_types": ["organic"],
            "order_by": ["keyword_data.keyword_info.search_volume,desc"]
        }])


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================
def clean_domain(url: str) -> str:
    url = url.strip().lower()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    domain = re.sub(r'^www\.', '', domain)
    domain = domain.rstrip('/')
    return domain


def format_number(n) -> str:
    if n is None:
        return "N/A"
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def parse_keywords_from_response(data: dict) -> list:
    keywords = []
    tasks = data.get("tasks", [])
    if tasks and tasks[0].get("status_code") == 20000:
        result = tasks[0].get("result", [])
        items = result[0].get("items", []) if result else []
        for item in items:
            kw_data = item.get("keyword_data", {})
            kw_info = kw_data.get("keyword_info", {})
            serp_elem = item.get("ranked_serp_element", {}).get("serp_item", {})
            keyword = kw_data.get("keyword", "")
            volume = kw_info.get("search_volume", 0) or 0
            position = serp_elem.get("rank_group", 0) or 0
            url = serp_elem.get("url", "") or ""
            if keyword and volume > 0:
                keywords.append({
                    "keyword": keyword,
                    "volume": volume,
                    "position": position,
                    "url": url,
                    "difficulty": kw_info.get("keyword_difficulty", 0) or 0,
                    "cpc": kw_info.get("cpc", 0) or 0,
                })
    return keywords


def cluster_keywords_with_claude(api_key: str, keywords: list, domain: str) -> dict:
    """Claude regroupe les mots-clés en clusters thématiques."""
    client = anthropic.Anthropic(api_key=api_key)

    kw_list = "\n".join([
        f"- {kw['keyword']} (vol: {kw['volume']}, pos: {kw['position']})"
        for kw in keywords[:300]
    ])

    prompt = f"""Tu es un expert SEO senior spécialisé en stratégie de contenu et autorité topique.

Analyse ces mots-clés du domaine "{domain}" et regroupe-les en clusters thématiques (topics).

## Mots-clés à clustériser :
{kw_list}

## Instructions :

1. **Identifie 5 à 15 clusters thématiques** (ni trop large genre "SEO", ni trop fin genre chaque mot-clé isolé). Chaque cluster = un sujet sur lequel le site peut construire de l'autorité.

2. **Nomme chaque cluster** avec un label clair et concis (2-4 mots max).

3. **Assigne chaque mot-clé** à exactement un cluster.

4. **Pour chaque cluster, évalue** :
   - `pillar_potential` (0-100) : capacité à être un pilier de contenu (volume total + cohérence thématique)
   - `content_depth` : "faible" / "moyen" / "fort" — profondeur du contenu existant (basé sur le nombre de mots-clés et positions)
   - `strategic_note` : une phrase d'insight stratégique pour ce cluster

5. **Identifie 3-5 sous-thèmes manquants** (topics adjacents que le domaine DEVRAIT couvrir mais qui n'apparaissent pas dans les mots-clés). Ce sont des gaps d'autorité topique.

## FORMAT DE RÉPONSE (JSON strict) :
```json
{{
  "domain_topic": "Le sujet principal / niche du domaine en 2-3 mots",
  "clusters": [
    {{
      "name": "Nom du cluster",
      "keywords": ["mot-clé 1", "mot-clé 2"],
      "pillar_potential": 85,
      "content_depth": "fort",
      "strategic_note": "Insight stratégique sur ce cluster"
    }}
  ],
  "topic_gaps": [
    {{
      "topic": "Nom du sous-thème manquant",
      "reason": "Pourquoi ce topic est important pour l'autorité",
      "priority": "haute/moyenne/basse",
      "suggested_keywords": ["mot-clé suggéré 1", "mot-clé suggéré 2"]
    }}
  ],
  "strategic_summary": "Synthèse stratégique en 3-4 phrases : forces, faiblesses, recommandation prioritaire"
}}
```

RÈGLES :
- Chaque mot-clé doit apparaître dans exactement un cluster
- Les noms de clusters doivent être en français si les mots-clés sont en français
- Sois précis dans les strategic_notes, pas de généralités vides
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        return json.loads(json_match.group(0))
    return {}


def analyze_competitor_gaps(api_key: str, domain: str, domain_clusters: dict,
                            competitor_keywords: dict) -> dict:
    """Claude analyse les gaps entre le domaine et ses concurrents par cluster."""
    client = anthropic.Anthropic(api_key=api_key)

    cluster_summary = []
    for cl in domain_clusters.get("clusters", []):
        cluster_summary.append(
            f"- {cl['name']} ({len(cl['keywords'])} KWs, pillar: {cl['pillar_potential']}/100, depth: {cl['content_depth']})"
        )

    comp_summary = []
    for comp_domain, kws in competitor_keywords.items():
        top_kws = sorted(kws, key=lambda x: x["volume"], reverse=True)[:20]
        kw_str = ", ".join([f"{k['keyword']}({k['volume']})" for k in top_kws])
        comp_summary.append(f"### {comp_domain}\n{kw_str}")

    prompt = f"""Tu es un expert SEO spécialisé en stratégie de contenu et autorité topique.

Le domaine "{domain}" a les clusters thématiques suivants :
{chr(10).join(cluster_summary)}

Topics gaps déjà identifiés : {json.dumps([g['topic'] for g in domain_clusters.get('topic_gaps', [])], ensure_ascii=False)}

Voici les mots-clés principaux de ses concurrents :
{chr(10).join(comp_summary)}

## Mission :
Compare la couverture thématique de "{domain}" vs ses concurrents et identifie :

1. **Clusters où {domain} est fort** (leader) vs les concurrents
2. **Clusters où {domain} est faible** (en retard) — les concurrents couvrent mieux
3. **Topics que les concurrents couvrent mais pas {domain}** (vrais gaps compétitifs)
4. **Plan d'action priorisé** : quels clusters développer en priorité, quels articles créer

## FORMAT JSON strict :
```json
{{
  "competitive_position": {{
    "leader_topics": ["Topic 1", "Topic 2"],
    "lagging_topics": ["Topic 3"],
    "blind_spots": [
      {{
        "topic": "Nom du topic",
        "covered_by": ["concurrent1.com", "concurrent2.com"],
        "estimated_volume": 5000,
        "priority": "haute"
      }}
    ]
  }},
  "action_plan": [
    {{
      "priority": 1,
      "action": "Créer un pilier sur X",
      "cluster": "Nom du cluster",
      "expected_impact": "Fort — volume potentiel de X/mois",
      "content_ideas": ["Idée article 1", "Idée article 2", "Idée article 3"]
    }}
  ]
}}
```
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        return json.loads(json_match.group(0))
    return {}


def compute_cluster_metrics(cluster: dict, all_keywords: list) -> dict:
    """Calcule les métriques agrégées d'un cluster."""
    kw_set = set(cluster["keywords"])
    matched = [kw for kw in all_keywords if kw["keyword"] in kw_set]

    total_volume = sum(kw["volume"] for kw in matched)
    avg_position = (sum(kw["position"] for kw in matched) / len(matched)) if matched else 0
    top3_count = sum(1 for kw in matched if kw["position"] <= 3)
    top10_count = sum(1 for kw in matched if kw["position"] <= 10)
    urls = set(kw["url"] for kw in matched if kw["url"])
    avg_difficulty = (sum(kw["difficulty"] for kw in matched) / len(matched)) if matched else 0

    authority_score = 0
    if matched:
        position_factor = max(0, (30 - avg_position) / 30) * 40
        coverage_factor = min(len(matched) / 10, 1.0) * 30
        top3_factor = min(top3_count / max(len(matched), 1), 1.0) * 30
        authority_score = round(position_factor + coverage_factor + top3_factor, 1)

    return {
        "total_volume": total_volume,
        "keyword_count": len(matched),
        "avg_position": round(avg_position, 1),
        "top3_count": top3_count,
        "top10_count": top10_count,
        "unique_urls": len(urls),
        "avg_difficulty": round(avg_difficulty, 1),
        "authority_score": min(authority_score, 100),
    }


# =============================================================================
# CONSTANTES
# =============================================================================
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

AUTHORITY_COLORS = {
    "fort": "#22C55E",
    "moyen": "#F59E0B",
    "faible": "#EF4444",
}


# =============================================================================
# INTERFACE
# =============================================================================
st.title("🗺️ Topical Authority Map")
st.markdown("*Cartographie ton autorité thématique : clusters, couverture, gaps et plan d'action.*")

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
    anthropic_key = st.text_input(
        "Clé API Claude (Anthropic)",
        value=st.secrets.get("ANTHROPIC_API_KEY", ""),
        type="password"
    )

    st.divider()
    st.header("🗺️ Paramètres")

    target_url = st.text_input(
        "Domaine à analyser",
        placeholder="monsite.fr",
        help="Entre le domaine sans https:// ni www."
    )

    selected_location = st.selectbox("Pays cible", list(LOCATIONS.keys()), index=0)

    nb_keywords = st.slider("Nombre de mots-clés à analyser", 50, 500, 200, 50,
                            help="Plus = analyse plus fine mais plus lente et coûteuse")

    nb_competitors = st.slider("Concurrents à comparer", 2, 5, 3,
                               help="Nombre de concurrents pour l'analyse de gaps")

    comp_keywords_limit = st.slider("Mots-clés par concurrent", 50, 200, 100, 50)

    exclude_top = st.checkbox("Exclure les gros portails", value=True)

    st.divider()
    st.markdown("""
    ### Comment ça marche ?

    **Phase 1 — Collecte** (DataForSEO)
    1. Récupère les mots-clés organiques du domaine
    2. Identifie les principaux concurrents
    3. Récupère les mots-clés des concurrents

    **Phase 2 — Clustering IA** (Claude)
    4. Regroupe les mots-clés en clusters thématiques
    5. Évalue la profondeur et le potentiel de chaque cluster

    **Phase 3 — Analyse de gaps**
    6. Compare la couverture vs concurrents
    7. Identifie les blind spots thématiques
    8. Génère un plan d'action priorisé

    **Phase 4 — Visualisation**
    9. Treemap de l'autorité topique
    10. Radar comparatif par cluster
    """)

# ─── Validations ───
if not dataforseo_username or not dataforseo_password:
    st.warning("Configure tes identifiants DataForSEO dans la sidebar.")
    st.stop()

if not anthropic_key:
    st.warning("Configure ta clé API Claude (Anthropic) dans la sidebar.")
    st.stop()

if not target_url:
    st.info("Entre un domaine dans la sidebar pour lancer l'analyse.")
    st.stop()

domain = clean_domain(target_url)
st.markdown(f"**Domaine analysé** : `{domain}`")

if st.button("🚀 Lancer la cartographie topique", type="primary", use_container_width=True):
    api = DataForSEOLabs(dataforseo_username, dataforseo_password)
    loc = LOCATIONS[selected_location]

    with st.status("Analyse en cours...", expanded=True) as status:

        # =====================================================================
        # PHASE 1 : COLLECTE DES MOTS-CLÉS DU DOMAINE
        # =====================================================================
        st.write("🔍 **Phase 1** — Récupération des mots-clés du domaine...")
        try:
            kw_result = api.get_ranked_keywords(
                domain=domain,
                location_code=loc["code"],
                language_code=loc["lang"],
                limit=nb_keywords
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                st.error("Identifiants DataForSEO invalides.")
            else:
                st.error(f"Erreur API : {e}")
            st.stop()

        target_keywords = parse_keywords_from_response(kw_result)

        if not target_keywords:
            st.warning("Impossible de récupérer les mots-clés. Vérifie le domaine.")
            st.stop()

        total_volume = sum(kw["volume"] for kw in target_keywords)
        st.write(f"  ✅ {len(target_keywords)} mots-clés récupérés (volume total : {format_number(total_volume)})")

        # =====================================================================
        # PHASE 1b : IDENTIFICATION DES CONCURRENTS
        # =====================================================================
        st.write("🔍 Identification des concurrents organiques...")
        try:
            comp_result = api.get_competitors(
                domain=domain,
                location_code=loc["code"],
                language_code=loc["lang"],
                limit=20,
                exclude_top_domains=exclude_top
            )
        except Exception as e:
            st.error(f"Erreur API concurrents : {e}")
            st.stop()

        tasks = comp_result.get("tasks", [])
        competitor_domains = []
        if tasks and tasks[0].get("status_code") == 20000:
            result = tasks[0].get("result", [])
            items = result[0].get("items", []) if result else []
            for item in items:
                comp_dom = item.get("domain", "")
                if comp_dom and comp_dom != domain:
                    intersections = item.get("intersections", 0)
                    competitor_domains.append({
                        "domain": comp_dom,
                        "intersections": intersections,
                    })

        competitor_domains = competitor_domains[:nb_competitors]
        st.write(f"  ✅ {len(competitor_domains)} concurrents sélectionnés : {', '.join(c['domain'] for c in competitor_domains)}")

        # =====================================================================
        # PHASE 1c : MOTS-CLÉS DES CONCURRENTS
        # =====================================================================
        competitor_keywords = {}
        for i, comp in enumerate(competitor_domains):
            st.write(f"  📊 Récupération des mots-clés de `{comp['domain']}` ({i+1}/{len(competitor_domains)})...")
            try:
                comp_kw_result = api.get_keywords_for_domain(
                    domain=comp["domain"],
                    location_code=loc["code"],
                    language_code=loc["lang"],
                    limit=comp_keywords_limit
                )
                comp_kws = parse_keywords_from_response(comp_kw_result)
                competitor_keywords[comp["domain"]] = comp_kws
                st.write(f"    ✅ {len(comp_kws)} mots-clés")
            except Exception as e:
                st.write(f"    ⚠️ Erreur pour {comp['domain']}: {e}")
                competitor_keywords[comp["domain"]] = []

        # =====================================================================
        # PHASE 2 : CLUSTERING IA (CLAUDE)
        # =====================================================================
        st.write("🧠 **Phase 2** — Clustering thématique avec Claude...")

        try:
            clustering_result = cluster_keywords_with_claude(
                api_key=anthropic_key,
                keywords=target_keywords,
                domain=domain,
            )
        except Exception as e:
            st.error(f"Erreur Claude API : {e}")
            st.stop()

        if not clustering_result or not clustering_result.get("clusters"):
            st.error("Impossible de parser le clustering de Claude.")
            st.stop()

        clusters = clustering_result["clusters"]
        topic_gaps = clustering_result.get("topic_gaps", [])
        strategic_summary = clustering_result.get("strategic_summary", "")
        domain_topic = clustering_result.get("domain_topic", domain)

        st.write(f"  ✅ {len(clusters)} clusters identifiés, {len(topic_gaps)} gaps thématiques")

        # Calculer les métriques de chaque cluster
        for cluster in clusters:
            metrics = compute_cluster_metrics(cluster, target_keywords)
            cluster.update(metrics)

        # =====================================================================
        # PHASE 3 : ANALYSE GAPS CONCURRENTIELS
        # =====================================================================
        st.write("🔎 **Phase 3** — Analyse des gaps vs concurrents...")

        try:
            gap_analysis = analyze_competitor_gaps(
                api_key=anthropic_key,
                domain=domain,
                domain_clusters=clustering_result,
                competitor_keywords=competitor_keywords,
            )
        except Exception as e:
            st.warning(f"Analyse de gaps partielle : {e}")
            gap_analysis = {}

        competitive_position = gap_analysis.get("competitive_position", {})
        action_plan = gap_analysis.get("action_plan", [])

        st.write(f"  ✅ Analyse terminée — {len(action_plan)} actions recommandées")

        status.update(label="Cartographie terminée !", state="complete", expanded=False)

    # =====================================================================
    # AFFICHAGE DES RÉSULTATS
    # =====================================================================
    st.divider()

    # ── Résumé stratégique ──
    st.markdown(f'<div class="section-title">📋 Synthèse stratégique — {domain_topic}</div>', unsafe_allow_html=True)

    # KPIs globaux
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mots-clés analysés", len(target_keywords))
    col2.metric("Volume total", format_number(total_volume))
    col3.metric("Clusters", len(clusters))
    col4.metric("Gaps identifiés", len(topic_gaps) + len(competitive_position.get("blind_spots", [])))
    col5.metric("Actions prioritaires", len(action_plan))

    if strategic_summary:
        st.info(strategic_summary)

    st.divider()

    # ── Treemap de l'autorité topique ──
    st.markdown('<div class="section-title">🗺️ Carte d\'autorité topique</div>', unsafe_allow_html=True)
    st.caption("Taille = volume de recherche du cluster | Couleur = score d'autorité (vert = fort, rouge = faible)")

    treemap_data = []
    for cl in clusters:
        treemap_data.append({
            "cluster": cl["name"],
            "volume": cl["total_volume"],
            "authority": cl["authority_score"],
            "keywords": cl["keyword_count"],
            "avg_pos": cl["avg_position"],
            "top3": cl["top3_count"],
            "depth": cl.get("content_depth", "?"),
            "pillar": cl.get("pillar_potential", 0),
            "label": f"{cl['name']}<br>Vol: {format_number(cl['total_volume'])} | Auth: {cl['authority_score']}/100"
        })

    if treemap_data:
        df_tree = pd.DataFrame(treemap_data)
        fig_tree = px.treemap(
            df_tree,
            path=["cluster"],
            values="volume",
            color="authority",
            color_continuous_scale=["#EF4444", "#F59E0B", "#22C55E"],
            range_color=[0, 100],
            custom_data=["keywords", "avg_pos", "top3", "depth", "pillar"],
        )
        fig_tree.update_traces(
            textinfo="label+value",
            texttemplate="<b>%{label}</b><br>Vol: %{value:,.0f}<br>Auth: %{color:.0f}/100",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Volume: %{value:,.0f}<br>"
                "Autorité: %{color:.0f}/100<br>"
                "Mots-clés: %{customdata[0]}<br>"
                "Position moy: %{customdata[1]}<br>"
                "Top 3: %{customdata[2]}<br>"
                "Profondeur: %{customdata[3]}<br>"
                "Potentiel pilier: %{customdata[4]}/100"
                "<extra></extra>"
            ),
        )
        fig_tree.update_layout(
            height=550,
            margin=dict(t=30, b=10, l=10, r=10),
            coloraxis_colorbar=dict(title="Autorité", ticksuffix="/100"),
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()

    # ── Détail par cluster ──
    st.markdown('<div class="section-title">📊 Détail par cluster</div>', unsafe_allow_html=True)

    clusters_sorted = sorted(clusters, key=lambda c: c["authority_score"], reverse=True)

    for i, cl in enumerate(clusters_sorted):
        depth_emoji = {"fort": "🟢", "moyen": "🟡", "faible": "🔴"}.get(cl.get("content_depth", ""), "⚪")
        auth_emoji = "🟢" if cl["authority_score"] >= 60 else "🟡" if cl["authority_score"] >= 30 else "🔴"

        with st.expander(
            f"{auth_emoji} **{cl['name']}** — Autorité {cl['authority_score']}/100 | "
            f"Vol {format_number(cl['total_volume'])} | {cl['keyword_count']} KWs | "
            f"Profondeur {depth_emoji} {cl.get('content_depth', '?')}",
            expanded=(i < 3)
        ):
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric("Autorité", f"{cl['authority_score']}/100")
            mc2.metric("Volume total", format_number(cl["total_volume"]))
            mc3.metric("Mots-clés", cl["keyword_count"])
            mc4.metric("Pos. moyenne", f"{cl['avg_position']}")
            mc5.metric("Top 3", cl["top3_count"])
            mc6.metric("Potentiel pilier", f"{cl.get('pillar_potential', 0)}/100")

            if cl.get("strategic_note"):
                st.markdown(f"💡 **Insight :** {cl['strategic_note']}")

            # Tableau des mots-clés du cluster
            kw_set = set(cl["keywords"])
            cluster_kws = [kw for kw in target_keywords if kw["keyword"] in kw_set]
            cluster_kws.sort(key=lambda x: x["volume"], reverse=True)

            if cluster_kws:
                df_kw = pd.DataFrame(cluster_kws)[["keyword", "volume", "position", "difficulty", "url"]]
                df_kw.columns = ["Mot-clé", "Volume", "Position", "Difficulté", "URL"]
                st.dataframe(
                    df_kw,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Position": st.column_config.NumberColumn(format="%d"),
                        "Volume": st.column_config.NumberColumn(format="%d"),
                        "Difficulté": st.column_config.ProgressColumn(
                            min_value=0, max_value=100, format="%d"
                        ),
                    },
                    height=min(len(cluster_kws) * 35 + 38, 350),
                )

            # URLs uniques
            unique_urls = list(set(kw["url"] for kw in cluster_kws if kw.get("url")))
            if unique_urls:
                st.markdown(f"**Pages positionnées** ({len(unique_urls)}) :")
                for url in unique_urls[:10]:
                    st.markdown(f"  - `{url}`")

    st.divider()

    # ── Radar comparatif ──
    st.markdown('<div class="section-title">🕸️ Couverture par cluster vs concurrents</div>', unsafe_allow_html=True)
    st.caption("Compare le volume capturé par cluster. Le domaine principal est en gras.")

    # Calculer la couverture des concurrents par cluster
    cluster_names = [cl["name"] for cl in clusters_sorted[:10]]
    cluster_keyword_sets = {cl["name"]: set(cl["keywords"]) for cl in clusters_sorted[:10]}

    radar_fig = go.Figure()
    colors = ["#667eea", "#f093fb", "#4fd1c5", "#f6ad55", "#fc8181", "#68d391"]

    # Domaine principal
    domain_volumes = []
    for cname in cluster_names:
        cl = next((c for c in clusters_sorted if c["name"] == cname), None)
        domain_volumes.append(cl["total_volume"] if cl else 0)

    max_vol = max(domain_volumes) if domain_volumes else 1

    radar_fig.add_trace(go.Scatterpolar(
        r=[v / max_vol * 100 for v in domain_volumes],
        theta=cluster_names,
        fill="toself",
        name=domain,
        line=dict(color=colors[0], width=3),
        opacity=0.7
    ))

    # Concurrents
    for ci, (comp_domain, comp_kws) in enumerate(competitor_keywords.items()):
        comp_volumes = []
        for cname in cluster_names:
            kw_set = cluster_keyword_sets[cname]
            matched_vol = sum(kw["volume"] for kw in comp_kws if kw["keyword"] in kw_set)
            comp_volumes.append(matched_vol)

        radar_fig.add_trace(go.Scatterpolar(
            r=[v / max_vol * 100 for v in comp_volumes],
            theta=cluster_names,
            fill="toself",
            name=comp_domain,
            line=dict(color=colors[(ci + 1) % len(colors)]),
            opacity=0.4
        ))

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 110])),
        height=550,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    st.divider()

    # ── Gaps thématiques ──
    st.markdown('<div class="section-title">🚨 Gaps d\'autorité topique</div>', unsafe_allow_html=True)

    all_gaps = topic_gaps.copy()
    blind_spots = competitive_position.get("blind_spots", [])
    if blind_spots:
        for bs in blind_spots:
            all_gaps.append({
                "topic": bs.get("topic", ""),
                "reason": f"Couvert par {', '.join(bs.get('covered_by', []))}. Volume estimé : {format_number(bs.get('estimated_volume', 0))}",
                "priority": bs.get("priority", "moyenne"),
                "suggested_keywords": bs.get("suggested_keywords", []) if "suggested_keywords" in bs else [],
            })

    if all_gaps:
        for gap in all_gaps:
            priority_color = {"haute": "🔴", "moyenne": "🟡", "basse": "🟢"}.get(gap.get("priority", ""), "⚪")
            st.markdown(f"""
<div class="gap-card">
    <strong>{priority_color} {gap['topic']}</strong> — Priorité : {gap.get('priority', '?')}<br>
    {gap.get('reason', '')}<br>
    {f"<em>Mots-clés suggérés : {', '.join(gap.get('suggested_keywords', []))}</em>" if gap.get('suggested_keywords') else ""}
</div>
""", unsafe_allow_html=True)
    else:
        st.success("Aucun gap majeur identifié — bonne couverture topique !")

    st.divider()

    # ── Plan d'action ──
    st.markdown('<div class="section-title">🎯 Plan d\'action priorisé</div>', unsafe_allow_html=True)

    leader_topics = competitive_position.get("leader_topics", [])
    lagging_topics = competitive_position.get("lagging_topics", [])

    if leader_topics or lagging_topics:
        col_lead, col_lag = st.columns(2)
        with col_lead:
            st.markdown("**✅ Topics où tu es leader :**")
            for t in leader_topics:
                st.markdown(f"  - 🟢 {t}")
        with col_lag:
            st.markdown("**⚠️ Topics où tu es en retard :**")
            for t in lagging_topics:
                st.markdown(f"  - 🔴 {t}")
        st.markdown("")

    if action_plan:
        for action in action_plan:
            priority = action.get("priority", "?")
            priority_badge = {1: "🥇", 2: "🥈", 3: "🥉"}.get(priority, f"#{priority}")
            st.markdown(f"""
<div class="opportunity-card">
    <strong>{priority_badge} {action.get('action', '')}</strong><br>
    Cluster : {action.get('cluster', 'N/A')} | Impact attendu : {action.get('expected_impact', 'N/A')}<br>
    <em>Idées de contenu :</em>
    <ul>
        {''.join(f"<li>{idea}</li>" for idea in action.get('content_ideas', []))}
    </ul>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Aucun plan d'action généré.")

    st.divider()

    # ── Tableau récapitulatif & Export ──
    st.markdown('<div class="section-title">📋 Tableau récapitulatif des clusters</div>', unsafe_allow_html=True)

    df_summary = pd.DataFrame([{
        "Cluster": cl["name"],
        "Autorité": cl["authority_score"],
        "Volume total": cl["total_volume"],
        "Mots-clés": cl["keyword_count"],
        "Position moy.": cl["avg_position"],
        "Top 3": cl["top3_count"],
        "Top 10": cl["top10_count"],
        "Pages": cl["unique_urls"],
        "Difficulté moy.": cl["avg_difficulty"],
        "Potentiel pilier": cl.get("pillar_potential", 0),
        "Profondeur": cl.get("content_depth", "?"),
    } for cl in clusters_sorted])

    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Autorité": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f"
            ),
            "Potentiel pilier": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f"
            ),
            "Difficulté moy.": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f"
            ),
        }
    )

    # Export CSV clusters
    csv_clusters = df_summary.to_csv(index=False).encode("utf-8")

    # Export CSV détaillé (tous les mots-clés avec leur cluster)
    detailed_rows = []
    for cl in clusters_sorted:
        kw_set = set(cl["keywords"])
        for kw in target_keywords:
            if kw["keyword"] in kw_set:
                detailed_rows.append({
                    "Cluster": cl["name"],
                    "Mot-clé": kw["keyword"],
                    "Volume": kw["volume"],
                    "Position": kw["position"],
                    "Difficulté": kw["difficulty"],
                    "CPC": kw["cpc"],
                    "URL": kw["url"],
                    "Autorité cluster": cl["authority_score"],
                    "Potentiel pilier": cl.get("pillar_potential", 0),
                })
    df_detailed = pd.DataFrame(detailed_rows)
    csv_detailed = df_detailed.to_csv(index=False).encode("utf-8")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Export clusters (résumé)",
            data=csv_clusters,
            file_name=f"topical_authority_{domain}_{selected_location}.csv",
            mime="text/csv"
        )
    with col_dl2:
        st.download_button(
            "📥 Export détaillé (tous les KWs)",
            data=csv_detailed,
            file_name=f"topical_authority_detailed_{domain}_{selected_location}.csv",
            mime="text/csv"
        )

    # ── Coût API ──
    st.caption(f"Coût API DataForSEO : ${api.total_cost:.4f}")

st.caption("🗺️ Topical Authority Map | Ma Toolbox SEO")
