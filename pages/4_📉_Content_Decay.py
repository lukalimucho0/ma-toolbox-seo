"""
Content Decay Detector — Détecteur de déclin de contenu
Identifie automatiquement les contenus en déclin via Google Search Console,
classifie le type de déclin et produit des recommandations priorisées.
"""

import streamlit as st
from utils.auth import check_password
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CSS (cohérent avec le reste de l'app)
# ============================================================

def inject_css():
    st.markdown("""
    <style>
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
        .insight-box {
            background-color: #F0F9FF;
            border-left: 4px solid #0EA5E9;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        .warning-box {
            background-color: #FFFBEB;
            border-left: 4px solid #F59E0B;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        .critical-box {
            background-color: #FEF2F2;
            border-left: 4px solid #EF4444;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        .success-box {
            background-color: #F0FDF4;
            border-left: 4px solid #22C55E;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #667eea;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# Client GSC (réutilise le pattern du projet)
# ============================================================

def _build_gsc_service():
    """Construit le service GSC à partir des secrets Streamlit."""
    try:
        from googleapiclient.discovery import build

        # Service account (prioritaire)
        if "GSC_SERVICE_ACCOUNT" in st.secrets:
            from google.oauth2 import service_account
            sa_info = dict(st.secrets["GSC_SERVICE_ACCOUNT"])
            creds = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
            )
            return build("searchconsole", "v1", credentials=creds)

        # OAuth refresh token (fallback)
        from google.oauth2.credentials import Credentials
        client_id = st.secrets.get("GSC_CLIENT_ID", "")
        client_secret = st.secrets.get("GSC_CLIENT_SECRET", "")
        refresh_token = st.secrets.get("GSC_REFRESH_TOKEN", "")
        if client_id and client_secret and refresh_token:
            creds = Credentials(
                token=None,
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                token_uri="https://oauth2.googleapis.com/token",
            )
            return build("searchconsole", "v1", credentials=creds)
    except ImportError:
        logger.warning("google-api-python-client ou google-auth non installé")
    except Exception as e:
        logger.error(f"Erreur init GSC : {e}")
    return None


# ============================================================
# Collecte GSC avec pagination
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gsc_data(site_url: str, start_date: str, end_date: str,
                   dimensions: tuple, row_limit: int = 5000,
                   url_filter: str = "") -> Optional[List[dict]]:
    """
    Récupère les données GSC avec pagination automatique.
    dimensions passé en tuple pour être hashable par le cache.
    """
    service = _build_gsc_service()
    if not service:
        return None

    all_rows: List[dict] = []
    start_row = 0
    page_size = min(row_limit, 25000)  # max API = 25000

    while True:
        body: dict = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": list(dimensions),
            "rowLimit": page_size,
            "startRow": start_row,
            "dataState": "final",
        }

        # Filtre URL optionnel
        if url_filter:
            body["dimensionFilterGroups"] = [{
                "filters": [{
                    "dimension": "page",
                    "operator": "contains",
                    "expression": url_filter,
                }]
            }]

        try:
            resp = service.searchanalytics().query(
                siteUrl=site_url, body=body,
            ).execute()
        except Exception as e:
            logger.error(f"Erreur GSC query: {e}")
            return None if not all_rows else all_rows

        rows = resp.get("rows", [])
        if not rows:
            break

        all_rows.extend(rows)

        # Stop si on a atteint la limite demandée ou s'il n'y a plus de pages
        if len(rows) < page_size or len(all_rows) >= row_limit:
            break
        start_row += len(rows)

    return all_rows


# ============================================================
# Traitement des données
# ============================================================

def build_page_metrics(rows: List[dict]) -> pd.DataFrame:
    """Transforme les rows GSC (dimension=page) en DataFrame."""
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        keys = r.get("keys", [])
        data.append({
            "url": keys[0] if keys else "",
            "clicks": r.get("clicks", 0),
            "impressions": r.get("impressions", 0),
            "ctr": r.get("ctr", 0),
            "position": r.get("position", 0),
        })
    return pd.DataFrame(data)


def build_page_query_metrics(rows: List[dict]) -> pd.DataFrame:
    """Transforme les rows GSC (dimension=page,query) en DataFrame."""
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        keys = r.get("keys", [])
        data.append({
            "url": keys[0] if len(keys) > 0 else "",
            "query": keys[1] if len(keys) > 1 else "",
            "clicks": r.get("clicks", 0),
            "impressions": r.get("impressions", 0),
            "ctr": r.get("ctr", 0),
            "position": r.get("position", 0),
        })
    return pd.DataFrame(data)


def build_monthly_metrics(rows: List[dict]) -> pd.DataFrame:
    """Transforme les rows GSC (dimension=page,date) en DataFrame mensuel."""
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        keys = r.get("keys", [])
        data.append({
            "url": keys[0] if len(keys) > 0 else "",
            "date": keys[1] if len(keys) > 1 else "",
            "clicks": r.get("clicks", 0),
            "impressions": r.get("impressions", 0),
        })
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    # Agréger par mois et URL
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["url", "month"]).agg(
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    ).reset_index()
    monthly["month"] = monthly["month"].dt.to_timestamp()
    return monthly.sort_values(["url", "month"])


def classify_decay(monthly_clicks: List[int], delta_pct: float,
                   impressions_delta_pct: float, ctr_delta_pct: float,
                   lost_kw_count: int) -> str:
    """
    Classifie le type de déclin d'une page.
    monthly_clicks : liste des clicks par mois (du plus ancien au plus récent).
    """
    n = len(monthly_clicks)

    # --- Baisse de CTR : impressions stables/hausse mais CTR en baisse ---
    if impressions_delta_pct > -10 and ctr_delta_pct < -15:
        return "Baisse de CTR"

    # --- Perte de mots-clés ---
    if lost_kw_count >= 3 and delta_pct < -30:
        return "Perte de mots-clés"

    # --- Déclin progressif : régression linéaire sur les données mensuelles ---
    if n >= 4:
        x = np.arange(n, dtype=float)
        y = np.array(monthly_clicks, dtype=float)
        if y.sum() > 0:
            # Régression linéaire simple
            slope, _ = np.polyfit(x, y, 1)
            # Calculer combien de mois sont en baisse par rapport au précédent
            declines = sum(1 for i in range(1, n) if y[i] < y[i - 1])
            if slope < 0 and declines >= n * 0.6:
                return "Déclin progressif"

    # --- Chute brutale : la majorité de la perte sur 1-2 mois ---
    if n >= 3:
        y = np.array(monthly_clicks, dtype=float)
        diffs = np.diff(y)
        if len(diffs) > 0 and diffs.min() < 0:
            total_loss = abs(diffs[diffs < 0].sum())
            max_monthly_loss = abs(diffs.min())
            if total_loss > 0 and max_monthly_loss / total_loss > 0.6:
                return "Chute brutale"

    return "Chute brutale"  # défaut


def compute_decay_score(delta_clicks_abs: int, delta_pct: float,
                        monthly_clicks: List[int], position_p1: float) -> float:
    """
    Score de priorité de 0 à 100.
    - 40% : volume de trafic perdu (clicks absolus)
    - 20% : pourcentage de baisse
    - 20% : consistance du déclin
    - 20% : récupérabilité (position 4-20 = bonus)
    """
    # --- Volume perdu (40%) ---
    # Normaliser : max attendu ~500 clicks perdus → 40 pts
    vol_score = min(abs(delta_clicks_abs) / 500.0, 1.0) * 40

    # --- % de baisse (20%) ---
    pct_score = min(abs(delta_pct) / 100.0, 1.0) * 20

    # --- Consistance (20%) ---
    consist_score = 0
    if len(monthly_clicks) >= 3:
        y = np.array(monthly_clicks, dtype=float)
        diffs = np.diff(y)
        if len(diffs) > 0:
            declining_months = (diffs < 0).sum()
            consist_score = (declining_months / len(diffs)) * 20

    # --- Récupérabilité (20%) ---
    recup_score = 0
    if 4 <= position_p1 <= 20:
        recup_score = 20
    elif 1 <= position_p1 < 4:
        recup_score = 10
    elif 20 < position_p1 <= 50:
        recup_score = 5

    return round(vol_score + pct_score + consist_score + recup_score, 1)


def analyze_decay(df_p1: pd.DataFrame, df_p2: pd.DataFrame,
                  df_pq1: pd.DataFrame, df_pq2: pd.DataFrame,
                  df_monthly: pd.DataFrame,
                  min_decline_pct: float, min_impressions: int) -> pd.DataFrame:
    """
    Produit le DataFrame final de pages en déclin.
    P1 = période récente, P2 = période ancienne (référence).
    """
    if df_p1.empty and df_p2.empty:
        return pd.DataFrame()

    # Fusionner P1 et P2 par URL
    merged = pd.merge(
        df_p2, df_p1, on="url", how="left", suffixes=("_p2", "_p1"),
    )

    # Pages présentes en P2 mais absentes en P1 → delta = -100%
    merged["clicks_p1"] = merged["clicks_p1"].fillna(0).astype(int)
    merged["impressions_p1"] = merged["impressions_p1"].fillna(0).astype(int)
    merged["ctr_p1"] = merged["ctr_p1"].fillna(0)
    merged["position_p1"] = merged["position_p1"].fillna(100)

    merged["clicks_p2"] = merged["clicks_p2"].astype(int)
    merged["impressions_p2"] = merged["impressions_p2"].astype(int)

    # Filtrer : impressions P2 minimum
    merged = merged[merged["impressions_p2"] >= min_impressions].copy()

    if merged.empty:
        return pd.DataFrame()

    # Calculs de delta
    merged["delta_clicks"] = merged["clicks_p1"] - merged["clicks_p2"]
    merged["delta_clicks_pct"] = np.where(
        merged["clicks_p2"] > 0,
        (merged["delta_clicks"] / merged["clicks_p2"]) * 100,
        np.where(merged["clicks_p1"] > 0, 0, -100),
    )
    merged["delta_impressions_pct"] = np.where(
        merged["impressions_p2"] > 0,
        ((merged["impressions_p1"] - merged["impressions_p2"]) / merged["impressions_p2"]) * 100,
        0,
    )
    merged["delta_position"] = merged["position_p1"] - merged["position_p2"]
    merged["ctr_delta_pct"] = np.where(
        merged["ctr_p2"] > 0,
        ((merged["ctr_p1"] - merged["ctr_p2"]) / merged["ctr_p2"]) * 100,
        0,
    )

    # Ne garder que les pages en déclin au-delà du seuil
    decayed = merged[merged["delta_clicks_pct"] <= -min_decline_pct].copy()

    if decayed.empty:
        return pd.DataFrame()

    # --- Analyse des mots-clés par page ---
    lost_kw_map: Dict[str, List[dict]] = defaultdict(list)
    top_kw_p2_map: Dict[str, str] = {}

    if not df_pq2.empty:
        # Top keyword P2 par page
        top_kw = df_pq2.sort_values("clicks", ascending=False).groupby("url").first()
        top_kw_p2_map = top_kw["query"].to_dict()

        # Mots-clés perdus : présents en P2 mais absents ou fortement réduits en P1
        pq2_grouped = df_pq2.groupby(["url", "query"]).agg(
            clicks_p2=("clicks", "sum"),
        ).reset_index()
        pq1_grouped = df_pq1.groupby(["url", "query"]).agg(
            clicks_p1=("clicks", "sum"),
        ).reset_index() if not df_pq1.empty else pd.DataFrame(columns=["url", "query", "clicks_p1"])

        kw_merged = pd.merge(pq2_grouped, pq1_grouped, on=["url", "query"], how="left")
        kw_merged["clicks_p1"] = kw_merged["clicks_p1"].fillna(0)
        kw_merged["kw_delta"] = kw_merged["clicks_p1"] - kw_merged["clicks_p2"]

        # Requêtes qui ont perdu > 50% de clicks
        lost = kw_merged[
            (kw_merged["clicks_p2"] > 0) &
            (kw_merged["kw_delta"] / kw_merged["clicks_p2"] < -0.5)
        ]
        for _, row in lost.iterrows():
            lost_kw_map[row["url"]].append({
                "query": row["query"],
                "clicks_p2": int(row["clicks_p2"]),
                "clicks_p1": int(row["clicks_p1"]),
                "delta": int(row["kw_delta"]),
            })

    # --- Données mensuelles par page ---
    monthly_map: Dict[str, List[int]] = {}
    if not df_monthly.empty:
        for url, grp in df_monthly.groupby("url"):
            monthly_map[url] = grp.sort_values("month")["clicks"].tolist()

    # --- Construire le résultat final ---
    results = []
    for _, row in decayed.iterrows():
        url = row["url"]
        monthly_clicks = monthly_map.get(url, [])
        lost_kws = lost_kw_map.get(url, [])

        decay_type = classify_decay(
            monthly_clicks=monthly_clicks,
            delta_pct=row["delta_clicks_pct"],
            impressions_delta_pct=row["delta_impressions_pct"],
            ctr_delta_pct=row["ctr_delta_pct"],
            lost_kw_count=len(lost_kws),
        )

        decay_score = compute_decay_score(
            delta_clicks_abs=row["delta_clicks"],
            delta_pct=row["delta_clicks_pct"],
            monthly_clicks=monthly_clicks,
            position_p1=row["position_p1"],
        )

        results.append({
            "url": url,
            "decay_score": decay_score,
            "decay_type": decay_type,
            "clicks_p2": row["clicks_p2"],
            "clicks_p1": row["clicks_p1"],
            "delta_clicks": row["delta_clicks"],
            "delta_clicks_pct": round(row["delta_clicks_pct"], 1),
            "impressions_p2": row["impressions_p2"],
            "impressions_p1": row["impressions_p1"],
            "delta_impressions_pct": round(row["delta_impressions_pct"], 1),
            "position_p2": round(row["position_p2"], 1),
            "position_p1": round(row["position_p1"], 1),
            "delta_position": round(row["delta_position"], 1),
            "ctr_p2": row["ctr_p2"],
            "ctr_p1": row["ctr_p1"],
            "top_keyword_p2": top_kw_p2_map.get(url, ""),
            "lost_keywords": lost_kws,
            "lost_keywords_count": len(lost_kws),
            "monthly_clicks": monthly_clicks,
        })

    df_result = pd.DataFrame(results)
    if not df_result.empty:
        df_result = df_result.sort_values("decay_score", ascending=False).reset_index(drop=True)
    return df_result


# ============================================================
# Diagnostics automatiques (par règles, sans LLM)
# ============================================================

DECAY_TYPE_EMOJI = {
    "Déclin progressif": "📉",
    "Chute brutale": "⚡",
    "Perte de mots-clés": "🔑",
    "Baisse de CTR": "👆",
    "Saisonnalité possible": "📅",
}


def generate_diagnostic(row: pd.Series) -> str:
    """Génère un diagnostic texte automatique pour une page."""
    dtype = row["decay_type"]
    delta = abs(row["delta_clicks_pct"])
    lost = row["lost_keywords_count"]
    n_months = len(row["monthly_clicks"])

    if dtype == "Déclin progressif":
        return (
            f"Cette page perd du trafic de manière régulière depuis {n_months} mois "
            f"({delta:.0f}% de baisse). Le contenu a probablement besoin d'être "
            f"mis à jour et enrichi pour rester compétitif."
        )
    elif dtype == "Perte de mots-clés":
        top_lost = ""
        if row["lost_keywords"] and len(row["lost_keywords"]) > 0:
            top_lost = row["lost_keywords"][0]["query"]
        return (
            f"Cette page a perdu ses positions sur {lost} requêtes clés"
            + (f", notamment « {top_lost} »" if top_lost else "")
            + ". Vérifiez si un concurrent a publié du contenu plus récent ou "
              "plus complet sur ces sujets."
        )
    elif dtype == "Baisse de CTR":
        ctr_drop = abs(row["ctr_p1"] - row["ctr_p2"]) * 100
        return (
            f"Les impressions sont relativement stables mais le CTR a baissé "
            f"de {ctr_drop:.1f} points. Optimisez le title et la meta description. "
            f"Vérifiez aussi si de nouvelles SERP features (featured snippets, PAA) "
            f"captent les clicks."
        )
    elif dtype == "Chute brutale":
        return (
            f"Cette page a subi une chute soudaine de {delta:.0f}%. "
            f"Cela peut correspondre à une mise à jour d'algorithme ou un changement technique. "
            f"Vérifiez l'indexation, les redirections et les logs serveur."
        )
    return f"Déclin détecté : {delta:.0f}% de baisse de clicks."


# ============================================================
# Affichage : KPIs
# ============================================================

def display_kpis(df: pd.DataFrame):
    cols = st.columns(4)
    with cols[0]:
        st.metric("Pages en déclin", len(df))
    with cols[1]:
        total_lost = int(df["delta_clicks"].sum())
        st.metric("Trafic total perdu", f"{total_lost:,} clicks")
    with cols[2]:
        avg_pct = df["delta_clicks_pct"].mean()
        st.metric("Perte moyenne", f"{avg_pct:+.1f}%")
    with cols[3]:
        progressive = len(df[df["decay_type"] == "Déclin progressif"])
        st.metric("Déclins progressifs", progressive,
                  help="Type le plus actionnable — le contenu vieillit")


# ============================================================
# Affichage : Graphique de répartition
# ============================================================

def display_decay_distribution(df: pd.DataFrame):
    counts = df["decay_type"].value_counts().reset_index()
    counts.columns = ["Type de déclin", "Nombre"]
    fig = px.bar(
        counts, x="Nombre", y="Type de déclin",
        orientation="h", color="Type de déclin",
        color_discrete_map={
            "Déclin progressif": "#667eea",
            "Chute brutale": "#EF4444",
            "Perte de mots-clés": "#F59E0B",
            "Baisse de CTR": "#0EA5E9",
            "Saisonnalité possible": "#22C55E",
        },
    )
    fig.update_layout(
        title="Répartition par type de déclin",
        showlegend=False, height=300,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Affichage : Tableau principal
# ============================================================

def display_main_table(df: pd.DataFrame, url_search: str) -> pd.DataFrame:
    """Affiche le tableau principal et retourne le df filtré."""
    display = df.copy()

    # Filtre texte
    if url_search:
        display = display[display["url"].str.contains(url_search, case=False, na=False)]

    if display.empty:
        st.info("Aucune page ne correspond au filtre.")
        return display

    # Préparer les colonnes d'affichage
    table = pd.DataFrame({
        "URL": display["url"].apply(lambda u: u.split("//", 1)[-1] if "//" in u else u),
        "Score": display["decay_score"],
        "Type": display["decay_type"].map(
            lambda t: f"{DECAY_TYPE_EMOJI.get(t, '')} {t}"
        ),
        "Clicks P2": display["clicks_p2"],
        "Clicks P1": display["clicks_p1"],
        "Delta clicks": display["delta_clicks"],
        "Delta %": display["delta_clicks_pct"].apply(lambda x: f"{x:+.1f}%"),
        "Δ Impressions %": display["delta_impressions_pct"].apply(lambda x: f"{x:+.1f}%"),
        "Pos. P2": display["position_p2"],
        "Pos. P1": display["position_p1"],
        "Top KW (P2)": display["top_keyword_p2"],
        "KW perdus": display["lost_keywords_count"],
    })

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f",
            ),
        },
    )
    return display


# ============================================================
# Affichage : Vue détaillée d'une page
# ============================================================

def display_page_detail(row: pd.Series, df_pq1: pd.DataFrame, df_pq2: pd.DataFrame):
    """Affiche le détail d'une page sélectionnée."""
    url = row["url"]

    # --- Sparkline mensuelle ---
    if row["monthly_clicks"] and len(row["monthly_clicks"]) > 1:
        months = row["monthly_clicks"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(months) + 1)),
            y=months,
            mode="lines+markers",
            line=dict(color="#EF4444", width=3),
            marker=dict(size=6),
            name="Clicks / mois",
        ))
        fig.update_layout(
            title=f"Évolution mensuelle des clicks",
            xaxis_title="Mois (du plus ancien au plus récent)",
            yaxis_title="Clicks",
            template="plotly_white", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Diagnostic automatique ---
    diag = generate_diagnostic(row)
    box_class = "critical-box" if row["decay_score"] > 70 else (
        "warning-box" if row["decay_score"] > 40 else "insight-box"
    )
    emoji = DECAY_TYPE_EMOJI.get(row["decay_type"], "📊")
    st.markdown(
        f'<div class="{box_class}">{emoji} <strong>Diagnostic :</strong> {diag}</div>',
        unsafe_allow_html=True,
    )

    # --- Tableau des mots-clés impactés ---
    if row["lost_keywords"] and len(row["lost_keywords"]) > 0:
        st.subheader("Mots-clés impactés")
        kw_df = pd.DataFrame(row["lost_keywords"])
        kw_df = kw_df.rename(columns={
            "query": "Requête", "clicks_p2": "Clicks P2",
            "clicks_p1": "Clicks P1", "delta": "Delta",
        })
        kw_df = kw_df.sort_values("Delta")
        st.dataframe(kw_df, use_container_width=True, hide_index=True)

    # Aussi montrer tous les KW de cette page (P2 vs P1) pour contexte complet
    pq2_page = df_pq2[df_pq2["url"] == url].copy() if not df_pq2.empty else pd.DataFrame()
    pq1_page = df_pq1[df_pq1["url"] == url].copy() if not df_pq1.empty else pd.DataFrame()
    if not pq2_page.empty:
        merged_kw = pd.merge(
            pq2_page[["query", "clicks", "position"]],
            pq1_page[["query", "clicks", "position"]] if not pq1_page.empty
            else pd.DataFrame(columns=["query", "clicks", "position"]),
            on="query", how="left", suffixes=("_p2", "_p1"),
        )
        merged_kw["clicks_p1"] = merged_kw["clicks_p1"].fillna(0).astype(int)
        merged_kw["position_p1"] = merged_kw["position_p1"].fillna(0)
        merged_kw["delta_clicks"] = merged_kw["clicks_p1"] - merged_kw["clicks_p2"]
        merged_kw = merged_kw.sort_values("delta_clicks")

        with st.expander(f"Tous les mots-clés de cette page ({len(merged_kw)})", expanded=False):
            display_kw = merged_kw.rename(columns={
                "query": "Requête",
                "clicks_p2": "Clicks P2", "clicks_p1": "Clicks P1",
                "position_p2": "Pos. P2", "position_p1": "Pos. P1",
                "delta_clicks": "Delta clicks",
            })
            st.dataframe(display_kw, use_container_width=True, hide_index=True)


# ============================================================
# Export CSV
# ============================================================

def export_csv(df: pd.DataFrame) -> str:
    export = df.copy()
    # Convertir les listes en texte
    export["lost_keywords_list"] = export["lost_keywords"].apply(
        lambda kws: ", ".join(k["query"] for k in kws) if kws else ""
    )
    export["monthly_clicks_str"] = export["monthly_clicks"].apply(
        lambda m: " → ".join(str(x) for x in m) if m else ""
    )
    cols = [
        "url", "decay_score", "decay_type",
        "clicks_p2", "clicks_p1", "delta_clicks", "delta_clicks_pct",
        "impressions_p2", "impressions_p1", "delta_impressions_pct",
        "position_p2", "position_p1", "delta_position",
        "ctr_p2", "ctr_p1",
        "top_keyword_p2", "lost_keywords_count", "lost_keywords_list",
        "monthly_clicks_str",
    ]
    return export[[c for c in cols if c in export.columns]].to_csv(index=False)


# ============================================================
# Export Markdown
# ============================================================

def export_markdown(df: pd.DataFrame, site_url: str,
                    p1_start: str, p1_end: str,
                    p2_start: str, p2_end: str,
                    min_decline: float, min_impr: int) -> str:
    total_lost = int(df["delta_clicks"].sum())
    avg_pct = df["delta_clicks_pct"].mean()
    progressive = len(df[df["decay_type"] == "Déclin progressif"])

    md = f"""# Content Decay Report — {site_url}
*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*

---

## Méthodologie

| Paramètre | Valeur |
|-----------|--------|
| Période récente (P1) | {p1_start} → {p1_end} |
| Période référence (P2) | {p2_start} → {p2_end} |
| Seuil de déclin minimum | {min_decline:.0f}% |
| Impressions minimum (P2) | {min_impr} |

## Synthèse

| Métrique | Valeur |
|----------|--------|
| Pages en déclin | {len(df)} |
| Trafic total perdu | {total_lost:,} clicks |
| Perte moyenne | {avg_pct:+.1f}% |
| Déclins progressifs | {progressive} |

## Top 20 pages en déclin

| # | URL | Score | Type | Clicks P2→P1 | Delta % |
|---|-----|------:|------|-------------:|--------:|
"""
    for i, (_, row) in enumerate(df.head(20).iterrows(), 1):
        short_url = row["url"].split("//", 1)[-1] if "//" in row["url"] else row["url"]
        # Tronquer l'URL pour lisibilité
        if len(short_url) > 60:
            short_url = short_url[:57] + "..."
        md += (
            f"| {i} | {short_url} | {row['decay_score']:.0f} "
            f"| {row['decay_type']} | {row['clicks_p2']}→{row['clicks_p1']} "
            f"| {row['delta_clicks_pct']:+.1f}% |\n"
        )

    md += "\n## Diagnostics détaillés (Top 10)\n\n"
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        short_url = row["url"].split("//", 1)[-1] if "//" in row["url"] else row["url"]
        diag = generate_diagnostic(row)
        lost_kws = ", ".join(k["query"] for k in (row["lost_keywords"] or [])[:5])
        md += f"### {i}. {short_url}\n\n"
        md += f"**Score** : {row['decay_score']:.0f}/100 | "
        md += f"**Type** : {row['decay_type']} | "
        md += f"**Delta** : {row['delta_clicks_pct']:+.1f}%\n\n"
        md += f"> {diag}\n\n"
        if lost_kws:
            md += f"**Mots-clés perdus** : {lost_kws}\n\n"

    md += "\n---\n*Rapport généré par Content Decay Detector*\n"
    return md


# ============================================================
# Main
# ============================================================

def main():
    st.set_page_config(
        page_title="Content Decay Detector",
        page_icon="📉",
        layout="wide",
    )

    check_password()
    inject_css()

    st.markdown('<p class="main-header">📉 Content Decay Detector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Identifiez les contenus en déclin et priorisez '
        'les mises à jour — basé sur Google Search Console</p>',
        unsafe_allow_html=True,
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Google Search Console")
        service = _build_gsc_service()
        if service:
            st.success("GSC connecté")
        else:
            st.error(
                "GSC non configuré — Ajoutez GSC_SERVICE_ACCOUNT ou "
                "GSC_CLIENT_ID/SECRET/REFRESH_TOKEN dans vos secrets."
            )

    # ---- Session state ----
    if "decay_results" not in st.session_state:
        st.session_state.decay_results = None
    if "decay_pq1" not in st.session_state:
        st.session_state.decay_pq1 = pd.DataFrame()
    if "decay_pq2" not in st.session_state:
        st.session_state.decay_pq2 = pd.DataFrame()
    if "decay_config" not in st.session_state:
        st.session_state.decay_config = {}

    # ===== ÉTAPE 1 : Configuration =====
    st.markdown('<p class="section-title">Configuration</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        site_url = st.text_input(
            "Propriété Google Search Console",
            placeholder="https://www.example.com/ ou sc-domain:example.com",
        )

        period_mode = st.radio(
            "Mode de période", ["Simple", "Avancé"], horizontal=True,
        )

        if period_mode == "Simple":
            duration = st.selectbox("Durée de comparaison", [
                "3 derniers mois vs 3 mois précédents",
                "6 derniers mois vs 6 mois précédents",
                "12 derniers mois vs 12 mois précédents",
            ], index=1)
            months_map = {"3": 90, "6": 180, "12": 365}
            days = months_map[duration.split()[0]]
            ref_date = datetime.now() - timedelta(days=3)  # marge GSC
            p1_end = ref_date.date()
            p1_start = (ref_date - timedelta(days=days)).date()
            p2_end = (p1_start - timedelta(days=1)).date()
            p2_start = (p2_end - timedelta(days=days - 1)).date()
        else:
            st.markdown("**Période 1 (récente)**")
            p1c1, p1c2 = st.columns(2)
            with p1c1:
                p1_start = st.date_input("Début P1", value=date.today() - timedelta(days=183))
            with p1c2:
                p1_end = st.date_input("Fin P1", value=date.today() - timedelta(days=3))
            st.markdown("**Période 2 (référence)**")
            p2c1, p2c2 = st.columns(2)
            with p2c1:
                p2_start = st.date_input("Début P2", value=date.today() - timedelta(days=366))
            with p2c2:
                p2_end = st.date_input("Fin P2", value=date.today() - timedelta(days=184))

    with col2:
        min_decline = st.slider(
            "Seuil de déclin minimum (%)", 10, 50, 20,
            help="Seules les pages ayant perdu au moins ce % de clicks seront affichées.",
        )
        min_impressions = st.number_input(
            "Impressions minimum (P2)", min_value=0, value=100, step=50,
            help="Exclure les pages à très faible visibilité.",
        )
        url_filter = st.text_input(
            "Filtre d'URL (optionnel)", placeholder="/blog/",
            help="N'analyser que les URLs contenant ce pattern.",
        )

    # Validation
    can_launch = bool(site_url.strip() and service)
    if not service:
        st.info("Configurez votre accès GSC dans les secrets Streamlit.")

    # ===== ÉTAPE 2 : Collecte =====
    if st.button("🔍 Analyser le déclin", type="primary",
                 disabled=not can_launch, use_container_width=True):

        p1_s = str(p1_start)
        p1_e = str(p1_end)
        p2_s = str(p2_start)
        p2_e = str(p2_end)
        full_start = str(p2_start)  # Pour les données mensuelles

        total_steps = 5
        step = 0
        progress = st.progress(0)
        status = st.status("Collecte des données GSC...", expanded=True)

        def tick(msg):
            nonlocal step
            step += 1
            progress.progress(step / total_steps)
            status.write(f"✅ {msg}")

        # 2a. Données par page — P1 et P2
        tick("GSC : pages période récente (P1)")
        raw_p1 = fetch_gsc_data(site_url.strip(), p1_s, p1_e,
                                ("page",), 5000, url_filter.strip())

        tick("GSC : pages période référence (P2)")
        raw_p2 = fetch_gsc_data(site_url.strip(), p2_s, p2_e,
                                ("page",), 5000, url_filter.strip())

        # 2b. Données page+query — P1 et P2
        tick("GSC : pages + requêtes (P1 & P2)")
        raw_pq1 = fetch_gsc_data(site_url.strip(), p1_s, p1_e,
                                 ("page", "query"), 10000, url_filter.strip())
        raw_pq2 = fetch_gsc_data(site_url.strip(), p2_s, p2_e,
                                 ("page", "query"), 10000, url_filter.strip())

        # 2c. Données mensuelles
        tick("GSC : données mensuelles (tendances)")
        raw_monthly = fetch_gsc_data(site_url.strip(), full_start, p1_e,
                                     ("page", "date"), 25000, url_filter.strip())

        # 2d. Traitement
        tick("Analyse du déclin...")
        df_p1 = build_page_metrics(raw_p1)
        df_p2 = build_page_metrics(raw_p2)
        df_pq1 = build_page_query_metrics(raw_pq1)
        df_pq2 = build_page_query_metrics(raw_pq2)
        df_monthly = build_monthly_metrics(raw_monthly)

        if df_p2.empty:
            st.error(
                "Aucune donnée GSC trouvée pour la période de référence. "
                "Vérifiez la propriété GSC et les dates."
            )
        else:
            results = analyze_decay(
                df_p1, df_p2, df_pq1, df_pq2, df_monthly,
                min_decline, min_impressions,
            )
            st.session_state.decay_results = results
            st.session_state.decay_pq1 = df_pq1
            st.session_state.decay_pq2 = df_pq2
            st.session_state.decay_config = {
                "site_url": site_url.strip(),
                "p1_start": p1_s, "p1_end": p1_e,
                "p2_start": p2_s, "p2_end": p2_e,
                "min_decline": min_decline, "min_impressions": min_impressions,
            }

        progress.progress(1.0)
        status.update(label="Analyse terminée !", state="complete")

    # ===== ÉTAPE 3 : Résultats =====
    if st.session_state.decay_results is not None:
        df = st.session_state.decay_results
        df_pq1 = st.session_state.decay_pq1
        df_pq2 = st.session_state.decay_pq2
        cfg = st.session_state.decay_config

        if df.empty:
            st.markdown(
                '<div class="success-box">✅ <strong>Peu de pages en déclin détectées</strong> '
                'avec ces critères. Essayez de baisser le seuil ou d\'élargir la période.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<p class="section-title">Résultats</p>', unsafe_allow_html=True)

            # 3a. KPIs
            display_kpis(df)
            st.divider()

            # 3b. Répartition
            col_chart, col_empty = st.columns([2, 1])
            with col_chart:
                display_decay_distribution(df)

            # 3c. Tableau principal
            st.markdown('<p class="section-title">Pages en déclin</p>', unsafe_allow_html=True)
            url_search = st.text_input(
                "🔎 Filtrer les URLs", placeholder="Tapez pour filtrer...",
                key="url_search_filter",
            )
            filtered = display_main_table(df, url_search)

            # 3d. Vue détaillée
            if not filtered.empty:
                st.divider()
                st.markdown('<p class="section-title">Détail par page</p>', unsafe_allow_html=True)

                url_options = filtered["url"].tolist()
                short_options = [
                    u.split("//", 1)[-1] if "//" in u else u for u in url_options
                ]
                selected_idx = st.selectbox(
                    "Sélectionnez une page",
                    range(len(url_options)),
                    format_func=lambda i: short_options[i],
                )
                if selected_idx is not None:
                    selected_url = url_options[selected_idx]
                    page_row = df[df["url"] == selected_url].iloc[0]
                    display_page_detail(page_row, df_pq1, df_pq2)

            # ===== ÉTAPE 4 : Export =====
            st.divider()
            st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)

            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                csv_data = export_csv(df)
                st.download_button(
                    "📥 Exporter en CSV",
                    data=csv_data,
                    file_name=f"content_decay_{cfg.get('site_url', 'site').replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col_exp2:
                md_data = export_markdown(
                    df, cfg.get("site_url", ""),
                    cfg.get("p1_start", ""), cfg.get("p1_end", ""),
                    cfg.get("p2_start", ""), cfg.get("p2_end", ""),
                    cfg.get("min_decline", 20), cfg.get("min_impressions", 100),
                )
                st.download_button(
                    "📥 Exporter le rapport (Markdown)",
                    data=md_data,
                    file_name=f"content_decay_report_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
