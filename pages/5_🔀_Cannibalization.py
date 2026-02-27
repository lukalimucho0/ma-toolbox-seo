"""
Page 5 — Détecteur de Cannibalisation SEO
Identifie les pages qui se concurrencent sur les mêmes requêtes Google,
classe par sévérité et propose des actions concrètes.
Fonctionne avec GSC (obligatoire) + Ahrefs (optionnel, enrichissement).
"""

import streamlit as st
from utils.auth import check_password
import pandas as pd
import numpy as np
import json
import io
import re
import zipfile
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.express as px
import requests

# ─── Auth ────────────────────────────────────────────────────────────────
check_password()

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constantes ──────────────────────────────────────────────────────────
AHREFS_API_BASE = "https://api.ahrefs.com/v3"
AHREFS_AVAILABLE = bool(st.secrets.get("AHREFS_API_TOKEN", ""))

SEVERITY_LABELS = {
    "critical": "🔴 Critique",
    "high": "🟠 Élevée",
    "moderate": "🟡 Modérée",
    "low": "🟢 Faible",
}

PERIOD_OPTIONS = {
    "28 derniers jours": 28,
    "3 derniers mois": 90,
    "6 derniers mois": 180,
    "12 derniers mois": 365,
}


# ─── CSS ─────────────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════
#  GSC — Construction du service et collecte
# ═══════════════════════════════════════════════════════════════════

def _build_gsc_service():
    """Construit le service GSC (Service Account prioritaire, puis OAuth)."""
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials

        # Service Account (priorité)
        if "GSC_SERVICE_ACCOUNT" in st.secrets:
            sa_info = dict(st.secrets["GSC_SERVICE_ACCOUNT"])
            creds = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
            )
            return build("searchconsole", "v1", credentials=creds)

        # OAuth Refresh Token (fallback)
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
        logger.warning("google-api-python-client non installé")
    except Exception as e:
        logger.error(f"Erreur init GSC : {e}")
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gsc_data(
    site_url: str,
    start_date: str,
    end_date: str,
    dimensions: tuple = ("query", "page"),
    row_limit: int = 25000,
    url_filter: str = "",
) -> Optional[List[dict]]:
    """Récupère les données GSC avec pagination automatique."""
    service = _build_gsc_service()
    if not service:
        return None

    all_rows: List[dict] = []
    start_row = 0
    page_size = min(row_limit, 25000)

    while True:
        body: dict = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": list(dimensions),
            "rowLimit": page_size,
            "startRow": start_row,
            "dataState": "final",
        }
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
            logger.error(f"Erreur requête GSC : {e}")
            return None if not all_rows else all_rows

        rows = resp.get("rows", [])
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size or len(all_rows) >= row_limit:
            break
        start_row += len(rows)

    return all_rows


def parse_gsc_rows(rows: List[dict]) -> pd.DataFrame:
    """Transforme les rows GSC (dimensions query+page) en DataFrame."""
    data = []
    for r in rows:
        keys = r.get("keys", [])
        data.append({
            "query": keys[0] if len(keys) > 0 else "",
            "page": keys[1] if len(keys) > 1 else "",
            "clicks": r.get("clicks", 0),
            "impressions": r.get("impressions", 0),
            "ctr": r.get("ctr", 0.0),
            "position": r.get("position", 0.0),
        })
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════
#  Ahrefs — Client REST v3 (optionnel)
# ═══════════════════════════════════════════════════════════════════

class AhrefsAPI:
    """Client léger pour Ahrefs REST v3."""

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        url = f"{AHREFS_API_BASE}/{endpoint}"
        params["output"] = "json"
        try:
            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ahrefs [{endpoint}] : {e}")
            return None

    @staticmethod
    def _extract_rows(data: Optional[dict]) -> list:
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
        return []

    def get_domain_rating(self, target: str, date: str) -> Optional[dict]:
        return self._get("site-explorer/domain-rating", {
            "target": target, "date": date,
        })

    def get_backlinks_stats(self, target: str, date: str,
                            mode: str = "subdomains") -> Optional[dict]:
        return self._get("site-explorer/backlinks-stats", {
            "target": target, "date": date, "mode": mode,
        })

    def get_page_metrics(self, urls: List[str]) -> Dict[str, dict]:
        """Récupère les métriques par page via batch-analysis."""
        targets = [
            {"url": u, "mode": "exact", "protocol": "both"}
            for u in urls[:100]  # Limiter à 100 URLs
        ]
        params = {
            "select": ["url_rating", "backlinks", "refdomains", "organic_traffic"],
            "targets": targets,
        }
        try:
            resp = self.session.post(
                f"{AHREFS_API_BASE}/batch-analysis",
                json=params,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            rows = data if isinstance(data, list) else data.get("results", [])
            result = {}
            for row in rows:
                url = row.get("target", row.get("url", ""))
                if url:
                    result[url] = {
                        "ahrefs_traffic": row.get("organic_traffic", 0),
                        "ahrefs_backlinks": row.get("backlinks", 0),
                        "ahrefs_refdomains": row.get("refdomains", 0),
                        "ahrefs_ur": row.get("url_rating", 0),
                    }
            return result
        except Exception as e:
            logger.error(f"Ahrefs batch-analysis : {e}")
            return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_domain_rating(token: str, target: str, date: str):
    return AhrefsAPI(token).get_domain_rating(target, date)


@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_page_metrics(token: str, urls: tuple) -> Dict[str, dict]:
    return AhrefsAPI(token).get_page_metrics(list(urls))


# ═══════════════════════════════════════════════════════════════════
#  Détection de cannibalisation — Scoring & Classification
# ═══════════════════════════════════════════════════════════════════

def compute_position_proximity_score(positions: List[float]) -> float:
    """Score 0-100 basé sur la proximité des positions (30% du total)."""
    if len(positions) < 2:
        return 0.0
    pos_range = max(positions) - min(positions)
    if pos_range <= 5:
        return 100.0
    elif pos_range <= 10:
        return 70.0
    elif pos_range <= 20:
        return 50.0
    else:
        return 30.0


def compute_click_distribution_score(clicks: List[int]) -> float:
    """Score 0-100 basé sur la répartition des clicks (30% du total).
    Plus les clicks sont répartis équitablement, pire c'est."""
    sorted_clicks = sorted(clicks, reverse=True)
    top = sorted_clicks[0]
    if top == 0:
        return 50.0  # Pas de clicks : considéré comme moyennement grave
    second = sorted_clicks[1] if len(sorted_clicks) > 1 else 0
    ratio = second / top  # 0 = un winner clair, 1 = répartition égale
    return ratio * 100.0


def compute_volume_score(total_impressions: int) -> float:
    """Score 0-100 basé sur le volume d'impressions (25% du total).
    Normalisation logarithmique."""
    if total_impressions <= 0:
        return 0.0
    # log(200) ≈ 5.3, log(1000) ≈ 6.9, log(10000) ≈ 9.2
    log_val = math.log(max(total_impressions, 1))
    # Normaliser entre 50 et 10000 impressions
    score = (log_val - math.log(50)) / (math.log(10000) - math.log(50)) * 100
    return max(0.0, min(100.0, score))


def compute_url_count_score(num_urls: int) -> float:
    """Score 0-100 basé sur le nombre d'URLs en concurrence (15% du total)."""
    if num_urls >= 4:
        return 100.0
    elif num_urls == 3:
        return 70.0
    else:
        return 40.0


def compute_severity_score(positions: List[float], clicks: List[int],
                           total_impressions: int, num_urls: int) -> float:
    """Score global de sévérité 0-100."""
    s1 = compute_position_proximity_score(positions) * 0.30
    s2 = compute_click_distribution_score(clicks) * 0.30
    s3 = compute_volume_score(total_impressions) * 0.25
    s4 = compute_url_count_score(num_urls) * 0.15
    return round(s1 + s2 + s3 + s4, 1)


def severity_label(score: float) -> str:
    """Retourne le label de sévérité à partir du score."""
    if score >= 75:
        return "critical"
    elif score >= 50:
        return "high"
    elif score >= 25:
        return "moderate"
    return "low"


def classify_pattern(urls: List[str], positions: List[float],
                     ctrs: List[float]) -> str:
    """Identifie le type de cannibalisation via heuristiques."""
    # Pagination / paramètres
    param_re = re.compile(r'[\?&](page|sort|order|filter|p)=', re.I)
    page_re = re.compile(r'/page/\d+', re.I)
    if any(param_re.search(u) or page_re.search(u) for u in urls):
        return "Pagination ou paramètres"

    # Blog vs page commerciale
    blog_patterns = re.compile(r'/(blog|article|guide|actualite|news|magazine|ressource)s?/', re.I)
    commercial_patterns = re.compile(r'/(produit|service|solution|tarif|prix|offre|categorie|shop|product)s?/', re.I)
    has_blog = any(blog_patterns.search(u) for u in urls)
    has_commercial = any(commercial_patterns.search(u) for u in urls)
    if has_blog and has_commercial:
        return "Blog vs page commerciale"

    # Page profonde vs page principale
    depths = [u.rstrip('/').count('/') for u in urls]
    if max(depths) - min(depths) >= 2:
        return "Page profonde vs page principale"

    # Intent mixte (CTR très différents)
    if len(ctrs) >= 2:
        sorted_ctrs = sorted(ctrs, reverse=True)
        if sorted_ctrs[0] > 0 and sorted_ctrs[-1] / sorted_ctrs[0] < 0.3:
            return "Intent mixte"

    # Défaut : contenus similaires
    return "Même intent, contenus similaires"


def recommend_action(pattern: str, severity: str, winner_url: str,
                     loser_urls: List[str], query: str,
                     ahrefs_data: Dict[str, dict]) -> str:
    """Génère une recommandation concrète par cas."""
    winner_short = winner_url.split("//")[-1] if "//" in winner_url else winner_url
    losers_short = [u.split("//")[-1] if "//" in u else u for u in loser_urls]

    # Déterminer le meilleur URL par backlinks Ahrefs si dispo
    if ahrefs_data:
        best_bl = winner_url
        best_bl_count = ahrefs_data.get(winner_url, {}).get("ahrefs_backlinks", 0)
        for lu in loser_urls:
            bl = ahrefs_data.get(lu, {}).get("ahrefs_backlinks", 0)
            if bl > best_bl_count:
                best_bl = lu
                best_bl_count = bl
        if best_bl != winner_url:
            winner_short = best_bl.split("//")[-1]
            winner_url = best_bl

    if pattern == "Pagination ou paramètres":
        return (
            f"Ajouter une balise canonical sur les variantes vers {winner_short}. "
            f"Vérifier que les pages paginées/filtrées ne sont pas indexées inutilement."
        )
    elif pattern == "Blog vs page commerciale":
        return (
            f"Différencier les intents : optimiser la page commerciale pour l'intent transactionnel "
            f"et l'article de blog pour l'intent informationnel de '{query}'. "
            f"Ajouter un lien interne depuis le blog vers la page commerciale avec l'ancre '{query}'."
        )
    elif pattern == "Page profonde vs page principale":
        return (
            f"Renforcer le maillage interne vers {winner_short} sur l'ancre '{query}'. "
            f"Vérifier que la page secondaire ne cible pas la même requête principale."
        )
    elif pattern == "Intent mixte":
        return (
            f"Les pages semblent cibler des intents différents. "
            f"Différencier le contenu de chaque page pour cibler des requêtes distinctes. "
            f"Renforcer le maillage interne vers {winner_short} pour '{query}'."
        )
    else:
        # Même intent, contenus similaires → fusionner
        losers_str = ", ".join(losers_short[:3])
        action = (
            f"Fusionner le contenu de {losers_str} dans {winner_short}. "
            f"Mettre en place une 301 des anciennes pages vers la page consolidée. "
            f"Mettre à jour le maillage interne pour pointer vers {winner_short} "
            f"sur l'ancre '{query}'."
        )
        if ahrefs_data:
            w_bl = ahrefs_data.get(winner_url, {}).get("ahrefs_backlinks", 0)
            action += f" (page cible : {w_bl} backlinks)"
        return action


def detect_cannibalization(
    df: pd.DataFrame,
    min_impressions: int,
    min_urls: int,
    brand_keyword: str = "",
    ahrefs_data: Dict[str, dict] = {},
) -> List[dict]:
    """Détecte les cas de cannibalisation à partir du DataFrame GSC."""
    if df.empty:
        return []

    # Filtrer requêtes de marque si demandé
    if brand_keyword:
        brand_lower = brand_keyword.strip().lower()
        df = df[~df["query"].str.lower().str.contains(brand_lower, na=False)].copy()

    # Grouper par requête
    grouped = df.groupby("query")

    cases = []
    for query, group in grouped:
        if len(group) < min_urls:
            continue
        total_imp = int(group["impressions"].sum())
        if total_imp < min_impressions:
            continue

        total_clicks = int(group["clicks"].sum())
        num_urls = len(group)

        # Trier par clicks décroissants
        sorted_group = group.sort_values("clicks", ascending=False)
        urls_list = sorted_group["page"].tolist()
        clicks_list = sorted_group["clicks"].astype(int).tolist()
        impressions_list = sorted_group["impressions"].astype(int).tolist()
        ctr_list = sorted_group["ctr"].tolist()
        pos_list = sorted_group["position"].tolist()

        # Construire les détails par URL
        url_details = []
        for i, row in enumerate(sorted_group.itertuples(index=False)):
            detail = {
                "url": row.page,
                "clicks": int(row.clicks),
                "impressions": int(row.impressions),
                "ctr": round(row.ctr, 4),
                "position": round(row.position, 1),
            }
            # Enrichissement Ahrefs
            if row.page in ahrefs_data:
                detail.update(ahrefs_data[row.page])
            url_details.append(detail)

        # Scoring
        score = compute_severity_score(pos_list, clicks_list, total_imp, num_urls)
        sev = severity_label(score)

        # Pattern & action
        pattern = classify_pattern(urls_list, pos_list, ctr_list)

        # Flag si 10+ URLs → problème d'indexation probable
        if num_urls >= 10:
            pattern = f"Problème d'indexation probable ({pattern})"

        winner_url = urls_list[0]
        loser_urls = urls_list[1:]

        action = recommend_action(
            pattern, sev, winner_url, loser_urls, str(query), ahrefs_data,
        )

        cases.append({
            "query": str(query),
            "total_impressions": total_imp,
            "total_clicks": total_clicks,
            "num_urls": num_urls,
            "urls": url_details,
            "severity": sev,
            "severity_score": score,
            "severity_label": SEVERITY_LABELS[sev],
            "pattern": pattern,
            "recommended_action": action,
            "winner_url": winner_url,
            "loser_urls": loser_urls,
        })

    # Trier par score décroissant
    cases.sort(key=lambda c: c["severity_score"], reverse=True)
    return cases


# ═══════════════════════════════════════════════════════════════════
#  Vues agrégées
# ═══════════════════════════════════════════════════════════════════

def build_url_summary(cases: List[dict]) -> pd.DataFrame:
    """Construit la vue groupée par URL."""
    url_data: Dict[str, dict] = {}
    for case in cases:
        winner = case["winner_url"]
        losers = set(case["loser_urls"])
        for ud in case["urls"]:
            u = ud["url"]
            if u not in url_data:
                url_data[u] = {
                    "url": u,
                    "nb_cas": 0,
                    "nb_winner": 0,
                    "nb_loser": 0,
                    "total_clicks": 0,
                    "actions": [],
                }
            url_data[u]["nb_cas"] += 1
            if u == winner:
                url_data[u]["nb_winner"] += 1
            if u in losers:
                url_data[u]["nb_loser"] += 1
            url_data[u]["total_clicks"] += ud["clicks"]
            url_data[u]["actions"].append(case["pattern"])

    rows = []
    for u, d in url_data.items():
        # Action dominante
        from collections import Counter
        action_counts = Counter(d["actions"])
        dominant = action_counts.most_common(1)[0][0] if action_counts else ""
        rows.append({
            "URL": d["url"],
            "Cas de cannibalisation": d["nb_cas"],
            "Fois winner": d["nb_winner"],
            "Fois loser": d["nb_loser"],
            "Clicks totaux": d["total_clicks"],
            "Pattern dominant": dominant,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Cas de cannibalisation", ascending=False)
    return df


# ═══════════════════════════════════════════════════════════════════
#  Diagnostic texte (sans LLM)
# ═══════════════════════════════════════════════════════════════════

def generate_diagnostic(case: dict) -> str:
    """Génère un diagnostic textuel par règles."""
    q = case["query"]
    n = case["num_urls"]
    total_clicks = case["total_clicks"]
    positions = [u["position"] for u in case["urls"]]
    pos_min, pos_max = min(positions), max(positions)
    pattern = case["pattern"]

    # Description de la situation
    diag = f"La requête **'{q}'** fait ranker **{n} pages différentes** de votre site. "

    # Positions
    if pos_max - pos_min <= 5:
        diag += (
            f"Les positions sont très proches ({', '.join(str(round(p)) for p in positions[:5])}), "
            f"ce qui indique que Google hésite fortement entre ces pages. "
        )
    elif pos_max - pos_min <= 15:
        diag += (
            f"Les positions sont relativement proches "
            f"(de {round(pos_min)} à {round(pos_max)}). "
        )
    else:
        diag += (
            f"L'écart de positions est important "
            f"(de {round(pos_min)} à {round(pos_max)}), "
            f"une page domine mais les autres diluent le signal. "
        )

    # Pattern
    if "Problème d'indexation" in pattern:
        diag += (
            f"Avec {n} URLs en concurrence, il s'agit probablement d'un **problème d'indexation** "
            f"(pages de tags, pagination, archives). "
        )
    elif pattern == "Blog vs page commerciale":
        diag += (
            "Un article de blog entre en concurrence avec une page commerciale : "
            "l'intent informationnel et transactionnel se mélangent. "
        )
    elif pattern == "Intent mixte":
        diag += (
            "Les CTR très différents entre les pages suggèrent que Google "
            "teste des intents différents. "
        )
    elif pattern == "Page profonde vs page principale":
        diag += (
            "Une page profonde rivalise avec une page de niveau supérieur. "
        )
    elif pattern == "Pagination ou paramètres":
        diag += (
            "La cannibalisation provient de variantes d'URL "
            "(pagination, paramètres de tri/filtre). "
        )
    else:
        diag += (
            "Les contenus semblent trop similaires et ciblent le même intent. "
        )

    # Impact
    diag += f"\n\n**Trafic total concerné** : {total_clicks} clicks sur la période analysée."

    # Estimation récupérable
    recoverable = round(total_clicks * 0.3)
    if recoverable > 0:
        diag += (
            f" En consolidant sur une seule URL, vous pourriez potentiellement "
            f"récupérer ~**{recoverable} clicks supplémentaires** (estimation +30%)."
        )

    return diag


# ═══════════════════════════════════════════════════════════════════
#  Exports
# ═══════════════════════════════════════════════════════════════════

def export_csv_zip(cases: List[dict], url_summary: pd.DataFrame) -> bytes:
    """Exporte deux CSV dans un ZIP."""
    # CSV 1 : par couple query/url
    rows_by_query = []
    for case in cases:
        for ud in case["urls"]:
            row = {
                "query": case["query"],
                "url": ud["url"],
                "clicks": ud["clicks"],
                "impressions": ud["impressions"],
                "ctr": ud["ctr"],
                "position": ud["position"],
                "severity_score": case["severity_score"],
                "severity": case["severity_label"],
                "num_urls": case["num_urls"],
                "pattern": case["pattern"],
                "recommended_action": case["recommended_action"],
                "is_winner": ud["url"] == case["winner_url"],
            }
            # Ahrefs si dispo
            for key in ("ahrefs_traffic", "ahrefs_backlinks", "ahrefs_refdomains"):
                if key in ud:
                    row[key] = ud[key]
            rows_by_query.append(row)

    df_query = pd.DataFrame(rows_by_query)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cannibalization_by_query.csv", df_query.to_csv(index=False))
        zf.writestr("cannibalization_by_url.csv", url_summary.to_csv(index=False))
    return buf.getvalue()


def export_markdown(cases: List[dict], url_summary: pd.DataFrame,
                    site_url: str, period: str, min_impressions: int) -> str:
    """Génère un rapport Markdown."""
    total = len(cases)
    critical = sum(1 for c in cases if c["severity"] == "critical")
    high = sum(1 for c in cases if c["severity"] == "high")
    total_clicks = sum(c["total_clicks"] for c in cases)
    recoverable = round(total_clicks * 0.3)

    md = f"# Rapport de Cannibalisation SEO\n\n"
    md += f"**Site** : {site_url}  \n"
    md += f"**Période** : {period}  \n"
    md += f"**Seuil d'impressions** : {min_impressions}  \n"
    md += f"**Date de génération** : {datetime.now().strftime('%d/%m/%Y %H:%M')}  \n\n"

    md += f"## Synthèse\n\n"
    md += f"- **{total}** cas de cannibalisation détectés\n"
    md += f"- **{critical}** cas critiques, **{high}** cas élevés\n"
    md += f"- **{total_clicks}** clicks concernés\n"
    md += f"- ~**{recoverable}** clicks potentiellement récupérables\n\n"

    # Répartition par sévérité
    md += "### Répartition par sévérité\n\n"
    for sev_key, sev_label in SEVERITY_LABELS.items():
        count = sum(1 for c in cases if c["severity"] == sev_key)
        if count:
            md += f"- {sev_label} : {count}\n"
    md += "\n"

    # Top 20 cas critiques
    md += "## Top 20 cas prioritaires\n\n"
    for i, case in enumerate(cases[:20], 1):
        md += f"### {i}. {case['query']}\n\n"
        md += f"**Sévérité** : {case['severity_label']} ({case['severity_score']}/100)  \n"
        md += f"**Pattern** : {case['pattern']}  \n"
        md += f"**URLs** ({case['num_urls']}) :  \n"
        for ud in case["urls"][:5]:
            marker = "**→ WINNER**" if ud["url"] == case["winner_url"] else "LOSER"
            md += (
                f"- `{ud['url']}` — pos {ud['position']}, "
                f"{ud['clicks']} clicks, CTR {ud['ctr']:.2%} [{marker}]\n"
            )
        md += f"\n**Diagnostic** :\n{generate_diagnostic(case)}\n\n"
        md += f"**Action recommandée** :\n{case['recommended_action']}\n\n---\n\n"

    # Top 10 URLs les plus cannibalisées
    md += "## Top 10 URLs les plus cannibalisées\n\n"
    md += "| URL | Cas | Winner | Loser | Clicks |\n"
    md += "|-----|-----|--------|-------|--------|\n"
    for _, row in url_summary.head(10).iterrows():
        md += (
            f"| {row['URL']} | {row['Cas de cannibalisation']} | "
            f"{row['Fois winner']} | {row['Fois loser']} | "
            f"{row['Clicks totaux']} |\n"
        )

    md += "\n\n---\n*Rapport généré par SEO Toolbox — Détecteur de Cannibalisation*\n"
    return md


# ═══════════════════════════════════════════════════════════════════
#  Affichage — KPIs, graphiques, tableaux
# ═══════════════════════════════════════════════════════════════════

def display_kpis(cases: List[dict]):
    """Affiche les KPIs de synthèse."""
    total = len(cases)
    critical = sum(1 for c in cases if c["severity"] == "critical")
    queries_impacted = len(set(c["query"] for c in cases))
    total_clicks = sum(c["total_clicks"] for c in cases)
    recoverable = round(total_clicks * 0.3)

    cols = st.columns(5)
    cols[0].metric("Cas détectés", total)
    cols[1].metric("Cas critiques 🔴", critical)
    cols[2].metric("Requêtes impactées", queries_impacted)
    cols[3].metric("Clicks concernés", f"{total_clicks:,}".replace(",", " "))
    cols[4].metric(
        "Clicks récupérables*",
        f"~{recoverable:,}".replace(",", " "),
        help="Estimation : +30% de clicks si consolidation sur une seule URL par requête.",
    )


def display_distribution_charts(cases: List[dict]):
    """Affiche les graphiques de répartition."""
    col1, col2 = st.columns(2)

    # Par sévérité
    sev_counts = {}
    for c in cases:
        label = c["severity_label"]
        sev_counts[label] = sev_counts.get(label, 0) + 1

    sev_colors = {
        SEVERITY_LABELS["critical"]: "#EF4444",
        SEVERITY_LABELS["high"]: "#F59E0B",
        SEVERITY_LABELS["moderate"]: "#EAB308",
        SEVERITY_LABELS["low"]: "#22C55E",
    }

    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=list(sev_counts.keys()),
            values=list(sev_counts.values()),
            hole=0.5,
            marker_colors=[sev_colors.get(k, "#94A3B8") for k in sev_counts.keys()],
        )])
        fig.update_layout(
            title="Répartition par sévérité",
            height=350,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Par pattern/action
    pattern_counts = {}
    for c in cases:
        p = c["pattern"]
        # Simplifier les patterns longs
        if "Problème d'indexation" in p:
            p = "Problème d'indexation"
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    with col2:
        fig2 = go.Figure(data=[go.Pie(
            labels=list(pattern_counts.keys()),
            values=list(pattern_counts.values()),
            hole=0.5,
        )])
        fig2.update_layout(
            title="Répartition par type de cannibalisation",
            height=350,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)


def display_main_table(cases: List[dict], severity_filter: List[str],
                       pattern_filter: List[str], search_text: str) -> pd.DataFrame:
    """Affiche le tableau principal filtré."""
    rows = []
    for c in cases:
        rows.append({
            "Requête": c["query"],
            "Sévérité": c["severity_label"],
            "Score": c["severity_score"],
            "URLs": c["num_urls"],
            "Impressions": c["total_impressions"],
            "Clicks": c["total_clicks"],
            "Pages en conflit": " | ".join(
                u.split("//")[-1][:60] for u in [ud["url"] for ud in c["urls"][:3]]
            ),
            "Pattern": c["pattern"],
            "Action": c["recommended_action"][:80] + "..." if len(c["recommended_action"]) > 80 else c["recommended_action"],
            "_severity_key": c["severity"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Filtres
    if severity_filter:
        sev_keys = []
        for sf in severity_filter:
            for k, v in SEVERITY_LABELS.items():
                if v == sf:
                    sev_keys.append(k)
        df = df[df["_severity_key"].isin(sev_keys)]

    if pattern_filter:
        df = df[df["Pattern"].isin(pattern_filter)]

    if search_text:
        mask = (
            df["Requête"].str.contains(search_text, case=False, na=False) |
            df["Pages en conflit"].str.contains(search_text, case=False, na=False)
        )
        df = df[mask]

    # Retirer la colonne interne
    df = df.drop(columns=["_severity_key"])

    return df


def display_page_detail(case: dict, all_cases: List[dict]):
    """Affiche la vue détaillée d'un cas."""
    st.markdown(f'<div class="section-title">{case["query"]}</div>', unsafe_allow_html=True)

    # Badge sévérité
    st.markdown(
        f'**{case["severity_label"]}** — Score : **{case["severity_score"]}/100** '
        f'— Pattern : **{case["pattern"]}**'
    )

    # Tableau comparatif
    has_ahrefs = any("ahrefs_traffic" in ud for ud in case["urls"])
    table_data = []
    for ud in case["urls"]:
        row = {
            "URL": ud["url"].split("//")[-1] if "//" in ud["url"] else ud["url"],
            "Position": ud["position"],
            "Clicks": ud["clicks"],
            "Impressions": ud["impressions"],
            "CTR": f'{ud["ctr"]:.2%}',
            "Rôle": "✅ Winner" if ud["url"] == case["winner_url"] else "❌ Loser",
        }
        if has_ahrefs:
            row["Trafic Ahrefs"] = ud.get("ahrefs_traffic", "—")
            row["Backlinks"] = ud.get("ahrefs_backlinks", "—")
            row["Ref Domains"] = ud.get("ahrefs_refdomains", "—")
        table_data.append(row)
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Bar chart comparatif
    fig = go.Figure()
    urls_short = [
        ud["url"].split("/")[-1][:40] or ud["url"].split("/")[-2][:40]
        for ud in case["urls"]
    ]
    clicks_vals = [ud["clicks"] for ud in case["urls"]]
    colors = ["#667eea" if ud["url"] == case["winner_url"] else "#94A3B8"
              for ud in case["urls"]]
    fig.add_trace(go.Bar(
        y=urls_short,
        x=clicks_vals,
        orientation="h",
        marker_color=colors,
        text=clicks_vals,
        textposition="auto",
    ))
    fig.update_layout(
        title="Répartition des clicks par URL",
        height=max(200, len(case["urls"]) * 50),
        margin=dict(t=40, b=20, l=20, r=20),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Clicks",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Diagnostic
    st.markdown('<div class="section-title">Diagnostic</div>', unsafe_allow_html=True)
    diag = generate_diagnostic(case)
    st.markdown(f'<div class="insight-box">{diag}</div>', unsafe_allow_html=True)

    # Recommandation détaillée
    st.markdown('<div class="section-title">Recommandation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="warning-box">{case["recommended_action"]}</div>', unsafe_allow_html=True)

    # Autres requêtes cannibalisées par les mêmes URLs
    involved_urls = set(ud["url"] for ud in case["urls"])
    related = []
    for other in all_cases:
        if other["query"] == case["query"]:
            continue
        other_urls = set(ud["url"] for ud in other["urls"])
        overlap = involved_urls & other_urls
        if len(overlap) >= 2:
            related.append(other)

    if related:
        st.markdown(
            '<div class="section-title">Autres requêtes cannibalisées par les mêmes URLs</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="critical-box">'
            f'<strong>{len(related)} autres cas</strong> impliquent les mêmes pages — '
            f'cannibalisation récurrente, priorité élevée !'
            f'</div>',
            unsafe_allow_html=True,
        )
        related_rows = []
        for r in related[:20]:
            related_rows.append({
                "Requête": r["query"],
                "Sévérité": r["severity_label"],
                "Score": r["severity_score"],
                "Clicks": r["total_clicks"],
                "Pattern": r["pattern"],
            })
        st.dataframe(pd.DataFrame(related_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    inject_css()

    st.markdown('<div class="main-header">🔀 Détecteur de Cannibalisation SEO</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Identifiez les pages qui se concurrencent sur les mêmes requêtes '
        'et obtenez des recommandations concrètes pour consolider votre trafic.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Badge Ahrefs
    if AHREFS_AVAILABLE:
        st.markdown(
            '<div class="success-box">✅ <strong>Données Ahrefs activées</strong> — '
            'L\'analyse sera enrichie avec le trafic estimé, backlinks et referring domains par page.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="insight-box">ℹ️ <strong>Mode GSC uniquement</strong> — '
            'Ajoutez un token Ahrefs dans les secrets pour enrichir l\'analyse '
            '(backlinks, trafic estimé).</div>',
            unsafe_allow_html=True,
        )

    # ─── Étape 1 : Configuration ────────────────────────────────
    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)

    with st.form("config_form"):
        gsc_property = st.text_input(
            "Propriété Google Search Console",
            placeholder="https://www.example.com/ ou sc-domain:example.com",
            help="Le format doit correspondre exactement à votre propriété GSC.",
        )

        # Pré-remplir le domaine Ahrefs à partir de la propriété GSC
        ahrefs_domain_default = ""
        if gsc_property:
            clean = gsc_property.replace("sc-domain:", "").replace("https://", "").replace("http://", "")
            ahrefs_domain_default = clean.strip("/")

        if AHREFS_AVAILABLE:
            ahrefs_domain = st.text_input(
                "Domaine Ahrefs",
                value=ahrefs_domain_default,
                help="Domaine pour les requêtes Ahrefs (pré-rempli depuis la propriété GSC).",
            )
        else:
            ahrefs_domain = ""

        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox(
                "Période d'analyse",
                list(PERIOD_OPTIONS.keys()),
                index=1,  # 3 derniers mois par défaut
                help="Plus la période est longue, plus la détection est fiable.",
            )
        with col2:
            min_impressions = st.number_input(
                "Seuil d'impressions minimum par requête",
                min_value=10,
                max_value=10000,
                value=50,
                step=10,
                help="Exclut les requêtes à très faible volume.",
            )

        col3, col4 = st.columns(2)
        with col3:
            min_urls_choice = st.radio(
                "Nombre minimum d'URLs par requête",
                ["2+", "3+"],
                index=0,
                horizontal=True,
                help="3+ ne montre que les cas de cannibalisation sévères.",
            )
        with col4:
            url_filter = st.text_input(
                "Filtre d'URL (optionnel)",
                placeholder="/blog/, /produits/",
                help="Ciblez un sous-dossier spécifique.",
            )

        # Option exclure requêtes de marque
        brand_keyword = st.text_input(
            "Exclure les requêtes de marque (optionnel)",
            placeholder="Nom de votre marque",
            help="Les requêtes contenant ce mot seront exclues.",
        )

        submitted = st.form_submit_button("🔍 Détecter la cannibalisation", type="primary")

    if not submitted:
        return

    if not gsc_property:
        st.error("Veuillez saisir une propriété GSC.")
        return

    # ─── Étape 2 : Collecte ─────────────────────────────────────
    min_urls = 3 if min_urls_choice == "3+" else 2
    days = PERIOD_OPTIONS[period]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    progress = st.progress(0)
    status_container = st.status("Collecte des données en cours...", expanded=True)

    # 2a. Collecte GSC
    with status_container:
        st.write("📡 Récupération des données GSC (query + page)...")

    gsc_rows = fetch_gsc_data(
        site_url=gsc_property,
        start_date=start_date,
        end_date=end_date,
        dimensions=("query", "page"),
        row_limit=25000,
        url_filter=url_filter,
    )

    if gsc_rows is None:
        progress.empty()
        status_container.update(label="Erreur", state="error")
        st.error(
            "Impossible de récupérer les données GSC. Vérifiez :\n"
            "- Le format de la propriété (ex: `https://www.example.com/` ou `sc-domain:example.com`)\n"
            "- Que le compte de service a accès à cette propriété\n"
            "- Que les credentials GSC sont configurées dans les secrets"
        )
        return

    if not gsc_rows:
        progress.empty()
        status_container.update(label="Aucune donnée", state="error")
        st.warning("Aucune donnée retournée par GSC pour cette période et cette propriété.")
        return

    progress.progress(0.3)
    with status_container:
        st.write(f"✅ {len(gsc_rows)} lignes GSC récupérées")

    # Parser les données
    df = parse_gsc_rows(gsc_rows)
    progress.progress(0.4)

    with status_container:
        st.write(f"📊 {df['query'].nunique()} requêtes uniques, {df['page'].nunique()} pages uniques")

    # 2c. Détection de cannibalisation (avant Ahrefs pour identifier les pages à enrichir)
    with status_container:
        st.write("🔍 Détection des cas de cannibalisation...")

    cases_preliminary = detect_cannibalization(
        df, min_impressions, min_urls, brand_keyword,
    )
    progress.progress(0.6)

    # 2b. Enrichissement Ahrefs (optionnel)
    ahrefs_data: Dict[str, dict] = {}
    if AHREFS_AVAILABLE and ahrefs_domain and cases_preliminary:
        token = st.secrets.get("AHREFS_API_TOKEN", "")
        with status_container:
            st.write("📡 Enrichissement Ahrefs (backlinks, trafic)...")

        # Collecter toutes les URLs uniques impliquées
        all_urls = set()
        for case in cases_preliminary:
            for ud in case["urls"]:
                all_urls.add(ud["url"])

        if all_urls:
            ahrefs_data = _ahrefs_page_metrics(token, tuple(sorted(all_urls)))
            with status_container:
                st.write(f"✅ Métriques Ahrefs récupérées pour {len(ahrefs_data)} pages")

        progress.progress(0.8)

    # Relancer la détection avec les données Ahrefs enrichies
    if ahrefs_data:
        cases = detect_cannibalization(
            df, min_impressions, min_urls, brand_keyword, ahrefs_data,
        )
    else:
        cases = cases_preliminary

    progress.progress(1.0)

    # ─── Résultat : peu ou pas de cas ────────────────────────────
    if not cases:
        status_container.update(label="Analyse terminée", state="complete")
        st.markdown(
            '<div class="success-box">'
            '🎉 <strong>Peu de cannibalisation détectée !</strong> '
            'Votre site semble bien structuré sur ce plan. '
            'Essayez de baisser le seuil d\'impressions ou d\'élargir la période pour une analyse plus fine.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    status_container.update(
        label=f"Analyse terminée — {len(cases)} cas détectés",
        state="complete",
    )

    # Stocker les résultats en session
    st.session_state["cannibal_cases"] = cases
    st.session_state["cannibal_df"] = df

    # ─── Étape 3 : Affichage ────────────────────────────────────
    st.markdown('<div class="section-title">Résultats</div>', unsafe_allow_html=True)

    # 3a. KPIs
    display_kpis(cases)

    # 3b. Distribution
    display_distribution_charts(cases)

    # 3c/3e. Onglets : Vue par requête / Vue par URL
    tab_query, tab_url = st.tabs(["📋 Par requête", "🔗 Par URL"])

    with tab_query:
        # Filtres
        st.markdown('<div class="section-title">Filtres</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            all_severities = sorted(set(c["severity_label"] for c in cases))
            severity_filter = st.multiselect("Sévérité", all_severities, default=[])
        with fc2:
            all_patterns = sorted(set(c["pattern"] for c in cases))
            pattern_filter = st.multiselect("Pattern", all_patterns, default=[])
        with fc3:
            search_text = st.text_input("Recherche (requête ou URL)", "")

        filtered_df = display_main_table(cases, severity_filter, pattern_filter, search_text)

        if filtered_df.empty:
            st.info("Aucun résultat pour ces filtres.")
        else:
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
                height=min(600, 40 + len(filtered_df) * 35),
            )

        # 3d. Vue détaillée
        st.markdown('<div class="section-title">Détail d\'un cas</div>', unsafe_allow_html=True)
        query_options = [c["query"] for c in cases]
        # Appliquer les mêmes filtres pour la sélection
        if severity_filter or pattern_filter or search_text:
            visible_queries = set(filtered_df["Requête"].tolist()) if not filtered_df.empty else set()
            query_options = [q for q in query_options if q in visible_queries]

        if query_options:
            selected_query = st.selectbox(
                "Sélectionnez une requête pour le détail",
                query_options,
                key="detail_select",
            )
            selected_case = next((c for c in cases if c["query"] == selected_query), None)
            if selected_case:
                display_page_detail(selected_case, cases)

    with tab_url:
        url_summary = build_url_summary(cases)
        if url_summary.empty:
            st.info("Aucune donnée URL à afficher.")
        else:
            st.markdown(
                '<div class="insight-box">'
                'Cette vue identifie les <strong>pages problématiques</strong> impliquées '
                'dans de nombreux cas de cannibalisation. Les pages fréquemment "loser" '
                'méritent une refonte complète.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                url_summary,
                use_container_width=True,
                hide_index=True,
                height=min(600, 40 + len(url_summary) * 35),
            )

    # ─── Étape 4 : Exports ──────────────────────────────────────
    st.markdown('<div class="section-title">Exports</div>', unsafe_allow_html=True)

    url_summary_for_export = build_url_summary(cases)

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_zip = export_csv_zip(cases, url_summary_for_export)
        st.download_button(
            "📥 Exporter en CSV (ZIP)",
            data=csv_zip,
            file_name=f"cannibalization_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
        )
    with exp_col2:
        md_report = export_markdown(
            cases, url_summary_for_export, gsc_property, period, min_impressions,
        )
        st.download_button(
            "📥 Exporter le rapport (Markdown)",
            data=md_report,
            file_name=f"cannibalization_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )


# ─── Point d'entrée ─────────────────────────────────────────────
main()
