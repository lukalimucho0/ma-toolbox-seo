"""
🎯 ANALYSE CONCURRENTIELLE SEO
================================
Détection des meilleurs concurrents organiques ET business d'un domaine.
Combine DataForSEO Labs (données SEO) + Claude (intelligence business).
"""

import streamlit as st
import requests
import base64
import pandas as pd
import plotly.graph_objects as go
from urllib.parse import urlparse
import anthropic
import json
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
    .stMetric > div { padding: 8px; }
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
        """Appel POST générique avec suivi des coûts."""
        response = self.session.post(f"{self.base_url}{endpoint}", json=payload)
        response.raise_for_status()
        data = response.json()
        self.total_cost += data.get("cost", 0)
        return data

    def get_competitors(self, domain: str, location_code: int, language_code: str,
                        limit: int = 50, exclude_top_domains: bool = True) -> dict:
        """Récupère les concurrents organiques d'un domaine."""
        return self._post("/dataforseo_labs/google/competitors_domain/live", [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "exclude_top_domains": exclude_top_domains,
            "item_types": ["organic"],
            "filters": ["metrics.organic.count", ">", 3],
            "order_by": ["metrics.organic.count,desc"]
        }])

    def get_ranked_keywords(self, domain: str, location_code: int, language_code: str,
                            limit: int = 50) -> dict:
        """Récupère les top mots-clés organiques d'un domaine."""
        return self._post("/dataforseo_labs/google/ranked_keywords/live", [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "item_types": ["organic"],
            "order_by": ["keyword_data.keyword_info.search_volume,desc"]
        }])

    def get_domain_metrics_bulk(self, domains: list, location_code: int, language_code: str) -> dict:
        """Récupère les concurrents d'un domaine pour en extraire ses métriques."""
        # On utilise ranked_keywords avec limit=1 juste pour avoir les métriques
        results = {}
        # Process in batches to avoid overloading
        for domain in domains:
            try:
                data = self._post("/dataforseo_labs/google/ranked_keywords/live", [{
                    "target": domain,
                    "location_code": location_code,
                    "language_code": language_code,
                    "limit": 1,
                    "item_types": ["organic"]
                }])
                tasks = data.get("tasks", [])
                if tasks and tasks[0].get("status_code") == 20000:
                    result = tasks[0].get("result", [])
                    if result:
                        metrics = result[0].get("metrics", {}).get("organic", {})
                        results[domain] = {
                            "total_keywords": metrics.get("count", 0),
                            "total_etv": metrics.get("etv", 0),
                            "estimated_paid_cost": metrics.get("estimated_paid_traffic_cost", 0),
                        }
            except Exception:
                pass
        return results


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


def format_number(n) -> str:
    """Formate un nombre pour l'affichage."""
    if n is None:
        return "N/A"
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def analyze_business_with_claude(api_key: str, domain: str, keywords: list,
                                  seo_competitors: list, nb_competitors: int) -> dict:
    """
    Utilise Claude pour :
    1. Identifier le coeur de métier du domaine
    2. Scorer chaque concurrent SEO sur la pertinence business
    3. Suggérer des concurrents business manquants
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Préparer la liste des mots-clés
    kw_list = "\n".join([f"- {kw['keyword']} (vol: {kw['volume']}, pos: {kw['position']})"
                         for kw in keywords[:40]])

    # Préparer la liste des concurrents SEO
    comp_list = "\n".join([f"- {c['domain']} ({c['intersections']} mots-clés communs, "
                           f"pos moy: {c['avg_position']}, trafic partagé: {c['shared_etv']:.0f})"
                           for c in seo_competitors[:30]])

    prompt = f"""Tu es un expert SEO senior. Analyse le domaine "{domain}" et ses concurrents.

## Mots-clés principaux du domaine (par volume de recherche) :
{kw_list}

## Concurrents SEO détectés (par chevauchement de mots-clés) :
{comp_list}

## Ta mission :

### 1. Identifie le coeur de métier
À partir du nom de domaine et de ses mots-clés, identifie précisément :
- Le secteur d'activité
- Le coeur de métier précis
- Le type de clientèle cible

### 2. Score chaque concurrent SEO
Pour chaque concurrent de la liste ci-dessus, attribue un score de pertinence BUSINESS de 0 à 100 :
- 100 = concurrent direct, même métier exact, même cible
- 70-99 = concurrent proche, métier similaire/adjacent
- 30-69 = acteur du même secteur large mais métier différent
- 0-29 = pas vraiment un concurrent business (portail, annuaire, media, agrégateur...)

### 3. Suggère des concurrents business manquants
Liste jusqu'à 10 domaines de vrais concurrents business qui ne sont PAS dans la liste SEO ci-dessus.
Ce sont des entreprises qui font le même métier mais qui peuvent être en retard en SEO.
Donne leur domaine exact (vérifié, pas inventé).

## FORMAT DE RÉPONSE OBLIGATOIRE (JSON strict) :
```json
{{
  "business_analysis": {{
    "sector": "...",
    "core_business": "...",
    "target_audience": "...",
    "summary": "Description en 2 phrases du positionnement business"
  }},
  "seo_competitors_scored": [
    {{"domain": "exemple.com", "business_score": 85, "reason": "Explication courte"}}
  ],
  "missing_business_competitors": [
    {{"domain": "exemple.com", "reason": "Explication courte de pourquoi c'est un concurrent direct"}}
  ]
}}
```

IMPORTANT :
- Pour missing_business_competitors, ne donne QUE des domaines réels et vérifiés, pas des domaines inventés.
- Le JSON doit être valide et parsable.
- Sois précis dans tes scores business : un annuaire ou un portail d'avis ne mérite PAS un score élevé.
- Les concurrents business directs (même activité, même cible) doivent avoir les scores les plus hauts.
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    # Extraire le JSON de la réponse
    response_text = response.content[0].text
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    else:
        # Tenter de parser directement
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group(0))
    return {}


def calculate_composite_score(seo_data: dict, business_score: float) -> float:
    """
    Score composite final (0-100).
    - 40% SEO (intersection mots-clés + position + trafic)
    - 60% Business (pertinence métier évaluée par Claude)

    Le business pèse plus lourd car c'est le problème identifié :
    un concurrent business en retard SEO doit quand même apparaître.
    """
    intersections = seo_data.get("intersections", 0)
    avg_position = seo_data.get("avg_position", 50)
    shared_etv = seo_data.get("shared_etv", 0)
    total_etv = seo_data.get("total_etv", 1)

    # Score SEO (0-100)
    position_score = max(0, (50 - avg_position) / 50 * 100) if avg_position <= 50 else 0
    traffic_ratio = min(shared_etv / max(total_etv, 1), 1.0) * 100

    seo_score = (position_score * 0.5) + (traffic_ratio * 0.5)
    seo_score = min(seo_score, 100)

    # Composite
    composite = (seo_score * 0.4) + (business_score * 0.6)
    return round(min(composite, 100), 1)


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
st.markdown("*Identifie tes vrais concurrents : ceux qui rivalisent en SEO **et** en business.*")

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
    st.header("🎯 Paramètres")

    target_url = st.text_input(
        "Domaine à analyser",
        placeholder="citydrop.fr",
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

    **Phase 1 — Données SEO** (DataForSEO)
    1. Récupère les mots-clés du domaine
    2. Trouve les domaines qui se positionnent dessus

    **Phase 2 — Intelligence business** (Claude)
    3. Identifie le coeur de métier
    4. Score chaque concurrent sur la pertinence business
    5. Suggère des concurrents business manquants

    **Phase 3 — Score composite**
    - 40% SEO (positions, trafic, intersection)
    - 60% Business (pertinence métier)
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

if st.button("🚀 Lancer l'analyse concurrentielle", type="primary", use_container_width=True):
    api = DataForSEOLabs(dataforseo_username, dataforseo_password)
    loc = LOCATIONS[selected_location]

    with st.status("Analyse en cours...", expanded=True) as status:

        # =====================================================================
        # PHASE 1 : DONNÉES SEO
        # =====================================================================
        st.write("🔍 **Phase 1** — Récupération des mots-clés du domaine...")
        try:
            kw_result = api.get_ranked_keywords(
                domain=domain,
                location_code=loc["code"],
                language_code=loc["lang"],
                limit=50
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                st.error("Identifiants DataForSEO invalides.")
            else:
                st.error(f"Erreur API : {e}")
            st.stop()

        # Parser les mots-clés
        target_keywords = []
        kw_tasks = kw_result.get("tasks", [])
        if kw_tasks and kw_tasks[0].get("status_code") == 20000:
            kw_items = kw_tasks[0].get("result", [{}])[0].get("items", []) if kw_tasks[0].get("result") else []
            for item in kw_items:
                kw_data = item.get("keyword_data", {})
                kw_info = kw_data.get("keyword_info", {})
                serp_elem = item.get("ranked_serp_element", {}).get("serp_item", {})
                target_keywords.append({
                    "keyword": kw_data.get("keyword", ""),
                    "volume": kw_info.get("search_volume", 0),
                    "position": serp_elem.get("rank_group", 0),
                })

        if not target_keywords:
            st.warning("Impossible de récupérer les mots-clés du domaine. Vérifie le domaine.")
            st.stop()

        st.write(f"  ✅ {len(target_keywords)} mots-clés récupérés")

        # Récupérer les concurrents SEO
        st.write("🔍 Récupération des concurrents SEO...")
        try:
            comp_result = api.get_competitors(
                domain=domain,
                location_code=loc["code"],
                language_code=loc["lang"],
                limit=50,
                exclude_top_domains=exclude_top
            )
        except Exception as e:
            st.error(f"Erreur API concurrents : {e}")
            st.stop()

        tasks = comp_result.get("tasks", [])
        if not tasks or tasks[0].get("status_code") != 20000:
            st.error("Erreur DataForSEO lors de la récupération des concurrents.")
            st.stop()

        task_result = tasks[0].get("result", [])
        items = task_result[0].get("items", []) if task_result else []
        total_count = task_result[0].get("total_count", 0) if task_result else 0

        seo_competitors = []
        for item in items:
            comp_domain = item.get("domain", "")
            if comp_domain == domain:
                continue
            metrics_organic = item.get("metrics", {}).get("organic", {})
            full_metrics = item.get("full_domain_metrics", {}).get("organic", {})
            seo_competitors.append({
                "domain": comp_domain,
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
            })

        st.write(f"  ✅ {total_count} concurrents SEO trouvés")

        # =====================================================================
        # PHASE 2 : INTELLIGENCE BUSINESS (CLAUDE)
        # =====================================================================
        st.write("🧠 **Phase 2** — Analyse business avec Claude...")

        try:
            claude_analysis = analyze_business_with_claude(
                api_key=anthropic_key,
                domain=domain,
                keywords=target_keywords,
                seo_competitors=seo_competitors,
                nb_competitors=nb_competitors
            )
        except Exception as e:
            st.error(f"Erreur Claude API : {e}")
            st.stop()

        if not claude_analysis:
            st.error("Impossible de parser la réponse de Claude.")
            st.stop()

        business_info = claude_analysis.get("business_analysis", {})
        scored_competitors = {c["domain"]: c for c in claude_analysis.get("seo_competitors_scored", [])}
        missing_competitors = claude_analysis.get("missing_business_competitors", [])

        st.write(f"  ✅ Business identifié : **{business_info.get('core_business', 'N/A')}**")
        st.write(f"  ✅ {len(missing_competitors)} concurrents business supplémentaires suggérés")

        # =====================================================================
        # PHASE 3 : ENRICHISSEMENT DES CONCURRENTS BUSINESS MANQUANTS
        # =====================================================================
        missing_domains = [c["domain"] for c in missing_competitors
                          if c["domain"] not in {sc["domain"] for sc in seo_competitors}]

        business_only_competitors = []
        if missing_domains:
            st.write("📊 **Phase 3** — Récupération des métriques des concurrents business manquants...")
            missing_metrics = api.get_domain_metrics_bulk(
                domains=missing_domains[:8],
                location_code=loc["code"],
                language_code=loc["lang"]
            )

            for mc in missing_competitors:
                mc_domain = mc["domain"]
                if mc_domain in {sc["domain"] for sc in seo_competitors}:
                    continue
                metrics = missing_metrics.get(mc_domain, {})
                business_only_competitors.append({
                    "domain": mc_domain,
                    "intersections": 0,
                    "avg_position": 0,
                    "shared_keywords": 0,
                    "shared_etv": 0,
                    "total_keywords": metrics.get("total_keywords", 0),
                    "total_etv": metrics.get("total_etv", 0),
                    "estimated_paid_cost": metrics.get("estimated_paid_cost", 0),
                    "pos_1": 0,
                    "pos_2_3": 0,
                    "pos_4_10": 0,
                    "source": "business",
                    "business_reason": mc.get("reason", ""),
                })
            st.write(f"  ✅ {len(business_only_competitors)} concurrents business enrichis")

        # =====================================================================
        # PHASE 4 : SCORING COMPOSITE & CLASSEMENT FINAL
        # =====================================================================
        st.write("🏆 **Calcul du classement final...**")

        all_competitors = []

        # Concurrents SEO (avec score business de Claude)
        for comp in seo_competitors:
            claude_data = scored_competitors.get(comp["domain"], {})
            business_score = claude_data.get("business_score", 20)
            business_reason = claude_data.get("reason", "")

            composite = calculate_composite_score(comp, business_score)

            all_competitors.append({
                **comp,
                "business_score": business_score,
                "business_reason": business_reason,
                "composite_score": composite,
                "source": "seo+business" if business_score >= 50 else "seo",
            })

        # Concurrents business uniquement
        for comp in business_only_competitors:
            # Score business élevé puisque suggéré par Claude comme concurrent direct
            business_score = 90
            composite = calculate_composite_score(comp, business_score)

            all_competitors.append({
                **comp,
                "business_score": business_score,
                "business_reason": comp.get("business_reason", ""),
                "composite_score": composite,
                "source": "business",
            })

        # Trier par score composite
        all_competitors.sort(key=lambda x: x["composite_score"], reverse=True)
        top_competitors = all_competitors[:nb_competitors]

        status.update(label="Analyse terminée !", state="complete", expanded=False)

    # =====================================================================
    # AFFICHAGE DES RÉSULTATS
    # =====================================================================
    st.divider()

    # ── Analyse business ──
    st.subheader(f"🏢 Analyse business de `{domain}`")
    col_biz1, col_biz2 = st.columns([1, 2])
    with col_biz1:
        st.markdown(f"""
        | | |
        |---|---|
        | **Secteur** | {business_info.get('sector', 'N/A')} |
        | **Coeur de métier** | {business_info.get('core_business', 'N/A')} |
        | **Cible** | {business_info.get('target_audience', 'N/A')} |
        """)
    with col_biz2:
        st.info(business_info.get('summary', ''))

    st.divider()

    # ── Top concurrents ──
    st.header(f"🏆 Top {nb_competitors} concurrents de `{domain}`")
    st.caption(f"Pays : {selected_location} — Score = 40% SEO + 60% Business")

    # Vue d'ensemble
    cols = st.columns(min(nb_competitors, 5))
    for i, comp in enumerate(top_competitors[:5]):
        with cols[i]:
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            source_icon = {"seo+business": "🎯", "seo": "📊", "business": "🏢"}.get(comp["source"], "")
            st.metric(
                label=f"{medals[i]} {comp['domain']}",
                value=f"{comp['composite_score']}/100",
                delta=f"{source_icon} Business: {comp['business_score']}/100"
            )

    if nb_competitors > 5:
        cols2 = st.columns(nb_competitors - 5)
        for i, comp in enumerate(top_competitors[5:]):
            with cols2[i]:
                st.metric(
                    label=f"{i+6}. {comp['domain']}",
                    value=f"{comp['composite_score']}/100",
                    delta=f"Business: {comp['business_score']}/100"
                )

    st.divider()

    # ── Détail par concurrent ──
    for i, comp in enumerate(top_competitors):
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        medal = medals[i] if i < len(medals) else f"#{i+1}"
        source_label = {"seo+business": "SEO + Business", "seo": "SEO uniquement", "business": "Business (hors radar SEO)"}.get(comp["source"], "")

        with st.expander(f"{medal} **{comp['domain']}** — Score final : {comp['composite_score']}/100 — _{source_label}_", expanded=(i < 3)):

            # Scores
            score_col1, score_col2, score_col3 = st.columns(3)
            with score_col1:
                st.markdown(f"### 🎯 Score final : **{comp['composite_score']}**/100")
            with score_col2:
                st.markdown(f"### 🏢 Business : **{comp['business_score']}**/100")
            with score_col3:
                seo_part = comp['composite_score'] - (comp['business_score'] * 0.6)
                seo_equiv = round(seo_part / 0.4, 1) if seo_part > 0 else 0
                st.markdown(f"### 📊 SEO : **{seo_equiv}**/100")

            # Raison business
            if comp.get("business_reason"):
                st.markdown(f"💡 **Pourquoi ce concurrent :** {comp['business_reason']}")

            # Métriques SEO
            if comp["source"] != "business":
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Mots-clés communs", format_number(comp["intersections"]))
                c2.metric("Position moy.", f"{comp['avg_position']}")
                c3.metric("Trafic partagé", format_number(comp["shared_etv"]))
                c4.metric("Trafic total", format_number(comp["total_etv"]))
                c5.metric("Mots-clés totaux", format_number(comp["total_keywords"]))

                # Graphique positions
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
                    **Données SEO :**
                    - **{comp['intersections']}** mots-clés en commun avec `{domain}`
                    - **{overlap_pct}%** de ses mots-clés chevauchent les tiens
                    - **{traffic_shared_pct}%** de son trafic vient de ces mots-clés partagés
                    - Valeur publicitaire estimée : **{format_number(comp['estimated_paid_cost'])}€/mois**
                    """)
            else:
                # Concurrent business uniquement
                st.markdown(f"""
                **⚠️ Ce concurrent n'apparaît pas dans le radar SEO** — il partage peu ou pas de mots-clés avec toi.
                Cela peut signifier :
                - Il est en retard sur le SEO
                - Il utilise des mots-clés différents pour le même métier
                - C'est une **opportunité** : regarde sa stratégie de contenu
                """)
                if comp["total_keywords"] > 0:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mots-clés totaux", format_number(comp["total_keywords"]))
                    c2.metric("Trafic total", format_number(comp["total_etv"]))
                    c3.metric("Valeur pub.", format_number(comp["estimated_paid_cost"]) + "€")

    # ── Tableau comparatif ──
    st.divider()
    st.subheader("📋 Tableau comparatif")

    df = pd.DataFrame([{
        "Rang": i + 1,
        "Domaine": c["domain"],
        "Score final": c["composite_score"],
        "Score Business": c["business_score"],
        "Source": {"seo+business": "SEO+Business", "seo": "SEO", "business": "Business"}.get(c["source"], ""),
        "Mots-clés communs": c["intersections"],
        "Position moy.": c["avg_position"],
        "Trafic partagé": round(c["shared_etv"]),
        "Trafic total": round(c["total_etv"]),
        "Mots-clés totaux": c["total_keywords"],
    } for i, c in enumerate(top_competitors)])

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score final": st.column_config.ProgressColumn(
                "Score final", min_value=0, max_value=100, format="%.1f"
            ),
            "Score Business": st.column_config.ProgressColumn(
                "Score Business", min_value=0, max_value=100, format="%.0f"
            ),
        }
    )

    # ── Graphique radar ──
    st.subheader("🕸️ Comparaison radar")

    max_intersections = max((c["intersections"] for c in top_competitors), default=1) or 1
    max_etv = max((c["shared_etv"] for c in top_competitors), default=1) or 1
    max_total_kw = max((c["total_keywords"] for c in top_competitors), default=1) or 1

    fig_radar = go.Figure()
    colors = ["#667eea", "#f093fb", "#4fd1c5", "#f6ad55", "#fc8181",
              "#68d391", "#b794f4", "#63b3ed", "#fbb6ce", "#c6f6d5"]

    for i, comp in enumerate(top_competitors):
        position_score = max(0, (50 - comp["avg_position"]) / 50 * 100) if comp["avg_position"] > 0 else 0

        fig_radar.add_trace(go.Scatterpolar(
            r=[
                comp["intersections"] / max_intersections * 100,
                position_score,
                comp["shared_etv"] / max_etv * 100 if max_etv else 0,
                comp["total_keywords"] / max_total_kw * 100 if max_total_kw else 0,
                comp["business_score"],
            ],
            theta=["Mots-clés communs", "Position SEO", "Trafic partagé",
                   "Mots-clés totaux", "Pertinence Business"],
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
    st.caption(f"Coût API DataForSEO : ${api.total_cost:.4f}")

st.caption("🎯 Analyse Concurrentielle SEO | Ma Toolbox SEO")
