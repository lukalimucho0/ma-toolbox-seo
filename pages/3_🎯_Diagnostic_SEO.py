"""
Diagnostic SEO Client — SEO Strategy Advisor
Agrège les données Ahrefs API v3, Google Search Console et Claude
pour produire un diagnostic stratégique complet et un plan d'action priorisé.
"""

import streamlit as st
from utils.auth import check_password
import requests
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
import anthropic
import io

# ============================================================
# Configuration & Constants
# ============================================================

AHREFS_API_BASE = "https://api.ahrefs.com/v3"

OBJECTIVES_OPTIONS = [
    "Augmenter le trafic organique",
    "Améliorer les conversions",
    "Visibilité marque",
    "Conquérir de nouveaux mots-clés",
    "Développer le netlinking",
    "Corriger les problèmes techniques",
]

CONSTRAINTS_OPTIONS = [
    "Budget limité",
    "Pas de netlinking",
    "Focus contenu uniquement",
    "Focus technique uniquement",
    "Site e-commerce",
    "Site éditorial",
    "Site local",
]

COUNTRY_OPTIONS = [
    ("FR", "France"),
    ("US", "États-Unis"),
    ("GB", "Royaume-Uni"),
    ("DE", "Allemagne"),
    ("ES", "Espagne"),
    ("IT", "Italie"),
    ("BE", "Belgique"),
    ("CH", "Suisse"),
    ("CA", "Canada"),
    ("NL", "Pays-Bas"),
    ("PT", "Portugal"),
    ("BR", "Brésil"),
]

# CTR attendu par position (données moyennes marché)
EXPECTED_CTR = {
    1: 0.35, 2: 0.17, 3: 0.11, 4: 0.08, 5: 0.065,
    6: 0.05, 7: 0.04, 8: 0.035, 9: 0.03, 10: 0.025,
    11: 0.02, 12: 0.018, 13: 0.016, 14: 0.014, 15: 0.012,
    16: 0.01, 17: 0.009, 18: 0.008, 19: 0.007, 20: 0.006,
}

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
# Ahrefs API Client (REST v3)
# ============================================================

class AhrefsAPI:
    """Client pour l'API Ahrefs REST v3."""

    def __init__(self, api_token: str):
        self.base_url = AHREFS_API_BASE
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        })

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        """GET request vers l'API Ahrefs v3."""
        url = f"{self.base_url}/{endpoint}"
        params["output"] = "json"
        try:
            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ahrefs API error [{endpoint}]: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Body: {e.response.text[:500]}")
            return None

    @staticmethod
    def _extract_rows(data: Optional[dict]) -> list:
        """Extrait la liste de rows d'une réponse Ahrefs (gère dict ou list)."""
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
        return []

    # --- Endpoints ---

    def get_domain_rating(self, target: str, date: str) -> Optional[dict]:
        return self._get("site-explorer/domain-rating", {
            "target": target, "date": date,
        })

    def get_metrics(self, target: str, date: str, mode: str = "subdomains") -> Optional[dict]:
        return self._get("site-explorer/metrics", {
            "target": target, "date": date, "mode": mode,
        })

    def get_backlinks_stats(self, target: str, date: str, mode: str = "subdomains") -> Optional[dict]:
        return self._get("site-explorer/backlinks-stats", {
            "target": target, "date": date, "mode": mode,
        })

    def get_top_pages(self, target: str, date: str, country: str = None,
                      limit: int = 20, mode: str = "subdomains") -> Optional[dict]:
        params = {
            "target": target, "date": date, "mode": mode,
            "select": "url,sum_traffic,top_keyword,top_keyword_best_position,keywords,value",
            "order_by": "sum_traffic:desc", "limit": limit,
        }
        if country:
            params["country"] = country
        return self._get("site-explorer/top-pages", params)

    def get_metrics_by_country(self, target: str, date: str,
                               mode: str = "subdomains") -> Optional[dict]:
        return self._get("site-explorer/metrics-by-country", {
            "target": target, "date": date, "mode": mode,
            "select": "country,org_traffic,org_keywords,org_keywords_1_3,org_cost",
        })

    def get_organic_keywords(self, target: str, date: str, country: str = "FR",
                             limit: int = 50, mode: str = "subdomains") -> Optional[dict]:
        return self._get("site-explorer/organic-keywords", {
            "target": target, "date": date, "mode": mode,
            "country": country, "limit": limit,
            "select": "keyword,volume,best_position,best_position_url,sum_traffic,keyword_difficulty,cpc",
            "order_by": "sum_traffic:desc",
        })

    def get_metrics_history(self, target: str, date_from: str,
                            date_to: str = None, mode: str = "subdomains") -> Optional[dict]:
        params = {
            "target": target, "date_from": date_from, "mode": mode,
            "history_grouping": "monthly",
            "select": "date,org_traffic,org_cost,paid_traffic",
        }
        if date_to:
            params["date_to"] = date_to
        return self._get("site-explorer/metrics-history", params)

    def get_referring_domains(self, target: str, limit: int = 20,
                              mode: str = "subdomains") -> Optional[dict]:
        return self._get("site-explorer/referring-domains", {
            "target": target, "mode": mode, "limit": limit,
            "select": "domain,domain_rating,dofollow_links,links_to_target,first_seen,traffic_domain",
            "order_by": "domain_rating:desc",
            "history": "live",
        })

    def get_organic_competitors(self, target: str, country: str, date: str,
                                limit: int = 10, mode: str = "subdomains") -> Optional[dict]:
        return self._get("site-explorer/organic-competitors", {
            "target": target, "country": country, "date": date,
            "mode": mode, "limit": limit,
            "select": "competitor_domain,keywords_common,keywords_competitor,keywords_target,share,traffic,domain_rating",
        })


# ============================================================
# Google Search Console API Client
# ============================================================

class GSCAPI:
    """Client pour l'API Google Search Console via google-api-python-client."""

    def __init__(self):
        self.service = None
        self._init_service()

    def _init_service(self):
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

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
                self.service = build("searchconsole", "v1", credentials=creds)
                return

            # Fallback : service account
            if "GSC_SERVICE_ACCOUNT" in st.secrets:
                from google.oauth2 import service_account
                sa_info = dict(st.secrets["GSC_SERVICE_ACCOUNT"])
                creds = service_account.Credentials.from_service_account_info(
                    sa_info,
                    scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
                )
                self.service = build("searchconsole", "v1", credentials=creds)
                return
        except ImportError:
            logger.warning("google-api-python-client ou google-auth non installé")
        except Exception as e:
            logger.error(f"GSC init error: {e}")
        self.service = None

    @property
    def is_configured(self) -> bool:
        return self.service is not None

    def list_properties(self) -> List[str]:
        if not self.service:
            return []
        try:
            resp = self.service.sites().list().execute()
            return [s["siteUrl"] for s in resp.get("siteEntry", [])]
        except Exception as e:
            logger.error(f"GSC list_properties: {e}")
            return []

    def search_analytics(self, site_url: str, dimensions: List[str],
                         start_date: str, end_date: str,
                         row_limit: int = 1000) -> Optional[List[dict]]:
        if not self.service:
            return None
        try:
            body = {
                "startDate": start_date,
                "endDate": end_date,
                "dimensions": dimensions,
                "rowLimit": row_limit,
                "dataState": "final",
            }
            resp = self.service.searchanalytics().query(
                siteUrl=site_url, body=body,
            ).execute()
            return resp.get("rows", [])
        except Exception as e:
            logger.error(f"GSC search_analytics: {e}")
            return None

    def performance_overview(self, site_url: str, days: int = 28) -> Optional[dict]:
        """Métriques agrégées (sans dimensions)."""
        if not self.service:
            return None
        end = datetime.now() - timedelta(days=3)
        start = end - timedelta(days=days)
        try:
            body = {
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate": end.strftime("%Y-%m-%d"),
                "dataState": "final",
            }
            resp = self.service.searchanalytics().query(
                siteUrl=site_url, body=body,
            ).execute()
            rows = resp.get("rows", [])
            return rows[0] if rows else None
        except Exception as e:
            logger.error(f"GSC performance_overview: {e}")
            return None


# ============================================================
# Fonctions de collecte cachées
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_domain_rating(token: str, target: str, date: str):
    return AhrefsAPI(token).get_domain_rating(target, date)

@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_metrics(token: str, target: str, date: str):
    return AhrefsAPI(token).get_metrics(target, date)

@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_backlinks_stats(token: str, target: str, date: str):
    return AhrefsAPI(token).get_backlinks_stats(target, date)

@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_top_pages(token: str, target: str, date: str, country: str, limit: int):
    return AhrefsAPI(token).get_top_pages(target, date, country, limit)

@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_metrics_by_country(token: str, target: str, date: str):
    return AhrefsAPI(token).get_metrics_by_country(target, date)

@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_organic_keywords(token: str, target: str, date: str, country: str, limit: int):
    return AhrefsAPI(token).get_organic_keywords(target, date, country, limit)

@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_metrics_history(token: str, target: str, date_from: str, date_to: str):
    return AhrefsAPI(token).get_metrics_history(target, date_from, date_to)

@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_referring_domains(token: str, target: str, limit: int):
    return AhrefsAPI(token).get_referring_domains(target, limit)

@st.cache_data(ttl=3600, show_spinner=False)
def _ahrefs_organic_competitors(token: str, target: str, country: str, date: str, limit: int):
    return AhrefsAPI(token).get_organic_competitors(target, country, date, limit)

@st.cache_data(ttl=3600, show_spinner=False)
def _gsc_analytics(site_url: str, dims: tuple, start: str, end: str, limit: int):
    return GSCAPI().search_analytics(site_url, list(dims), start, end, limit)

@st.cache_data(ttl=3600, show_spinner=False)
def _gsc_performance(site_url: str, days: int):
    return GSCAPI().performance_overview(site_url, days)


# ============================================================
# Orchestration de la collecte
# ============================================================

def collect_all_data(domain: str, competitors: List[str], gsc_property: str,
                     ahrefs_token: str, country: str = "FR") -> dict:
    """Collecte séquentielle de toutes les données avec progress bar."""

    today = datetime.now().strftime("%Y-%m-%d")
    date_12m_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Fenêtres GSC
    gsc_end = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    gsc_start_n = (datetime.now() - timedelta(days=31)).strftime("%Y-%m-%d")
    gsc_start_prev = (datetime.now() - timedelta(days=59)).strftime("%Y-%m-%d")
    gsc_end_prev = (datetime.now() - timedelta(days=32)).strftime("%Y-%m-%d")

    use_gsc = bool(gsc_property)
    total_steps = 9 + len([c for c in competitors if c.strip()]) + (6 if use_gsc else 0)
    step = 0

    data: dict = {
        "domain": domain,
        "competitors": competitors,
        "country": country,
        "collection_date": today,
        "ahrefs": {},
        "gsc": {},
        "errors": [],
    }

    progress = st.progress(0)
    status = st.status("Collecte des données en cours...", expanded=True)

    def tick(msg: str):
        nonlocal step
        step += 1
        progress.progress(min(step / total_steps, 1.0))
        status.write(f"{'✅'} {msg}")

    # ---- AHREFS ----

    tick("Ahrefs : Domain Rating")
    dr = _ahrefs_domain_rating(ahrefs_token, domain, today)
    data["ahrefs"]["domain_rating"] = dr
    if not dr:
        data["errors"].append("Ahrefs Domain Rating indisponible")

    tick("Ahrefs : Métriques générales")
    met = _ahrefs_metrics(ahrefs_token, domain, today)
    data["ahrefs"]["metrics"] = met
    if not met:
        data["errors"].append("Ahrefs Metrics indisponibles")

    tick("Ahrefs : Statistiques backlinks")
    bl = _ahrefs_backlinks_stats(ahrefs_token, domain, today)
    data["ahrefs"]["backlinks_stats"] = bl
    if not bl:
        data["errors"].append("Ahrefs Backlinks Stats indisponibles")

    tick("Ahrefs : Top pages par trafic")
    tp = _ahrefs_top_pages(ahrefs_token, domain, today, country, 20)
    data["ahrefs"]["top_pages"] = tp
    if not tp:
        data["errors"].append("Ahrefs Top Pages indisponibles")

    tick("Ahrefs : Répartition par pays")
    bc = _ahrefs_metrics_by_country(ahrefs_token, domain, today)
    data["ahrefs"]["metrics_by_country"] = bc
    if not bc:
        data["errors"].append("Ahrefs Metrics by Country indisponibles")

    tick("Ahrefs : Top keywords organiques")
    ok = _ahrefs_organic_keywords(ahrefs_token, domain, today, country, 50)
    data["ahrefs"]["organic_keywords"] = ok
    if not ok:
        data["errors"].append("Ahrefs Organic Keywords indisponibles")

    tick("Ahrefs : Historique 12 mois")
    hist = _ahrefs_metrics_history(ahrefs_token, domain, date_12m_ago, today)
    data["ahrefs"]["metrics_history"] = hist
    if not hist:
        data["errors"].append("Ahrefs Metrics History indisponible")

    tick("Ahrefs : Top referring domains")
    rd = _ahrefs_referring_domains(ahrefs_token, domain, 20)
    data["ahrefs"]["referring_domains"] = rd
    if not rd:
        data["errors"].append("Ahrefs Referring Domains indisponibles")

    tick("Ahrefs : Concurrents organiques")
    oc = _ahrefs_organic_competitors(ahrefs_token, domain, country, today, 10)
    data["ahrefs"]["organic_competitors"] = oc
    if not oc:
        data["errors"].append("Ahrefs Organic Competitors indisponibles")

    # Concurrents manuels
    data["ahrefs"]["competitors_data"] = {}
    for comp in competitors:
        comp = comp.strip()
        if not comp:
            continue
        tick(f"Ahrefs : Métriques concurrent {comp}")
        c_dr = _ahrefs_domain_rating(ahrefs_token, comp, today)
        c_met = _ahrefs_metrics(ahrefs_token, comp, today)
        c_bl = _ahrefs_backlinks_stats(ahrefs_token, comp, today)
        data["ahrefs"]["competitors_data"][comp] = {
            "domain_rating": c_dr, "metrics": c_met, "backlinks_stats": c_bl,
        }

    # ---- GSC ----
    if use_gsc:
        tick("GSC : Performance overview 28j")
        perf = _gsc_performance(gsc_property, 28)
        data["gsc"]["performance"] = perf
        if not perf:
            data["errors"].append("GSC Performance Overview indisponible")

        tick("GSC : Top requêtes")
        queries = _gsc_analytics(gsc_property, ("query",), gsc_start_n, gsc_end, 50)
        data["gsc"]["top_queries"] = queries
        if not queries:
            data["errors"].append("GSC Top Queries indisponibles")

        tick("GSC : Top pages")
        pages = _gsc_analytics(gsc_property, ("page",), gsc_start_n, gsc_end, 30)
        data["gsc"]["top_pages"] = pages
        if not pages:
            data["errors"].append("GSC Top Pages indisponibles")

        tick("GSC : Comparaison N vs N-1")
        data["gsc"]["current_period_queries"] = _gsc_analytics(
            gsc_property, ("query",), gsc_start_n, gsc_end, 500)
        data["gsc"]["prev_period_queries"] = _gsc_analytics(
            gsc_property, ("query",), gsc_start_prev, gsc_end_prev, 500)
        data["gsc"]["current_period_pages"] = _gsc_analytics(
            gsc_property, ("page",), gsc_start_n, gsc_end, 500)
        data["gsc"]["prev_period_pages"] = _gsc_analytics(
            gsc_property, ("page",), gsc_start_prev, gsc_end_prev, 500)

        tick("GSC : Répartition par device")
        devs = _gsc_analytics(gsc_property, ("device",), gsc_start_n, gsc_end, 10)
        data["gsc"]["devices"] = devs
        if not devs:
            data["errors"].append("GSC Devices indisponibles")

        tick("GSC : Données query+page (cannibalization)")
        qp = _gsc_analytics(gsc_property, ("query", "page"), gsc_start_n, gsc_end, 1000)
        data["gsc"]["query_page"] = qp

    progress.progress(1.0)
    status.update(label="Collecte terminée !", state="complete")
    return data


# ============================================================
# Calculs dérivés
# ============================================================

def calculate_quick_wins(gsc_queries: Optional[List[dict]]) -> pd.DataFrame:
    """Requêtes position 4-20 avec impressions > 100 et CTR sous-optimal."""
    if not gsc_queries:
        return pd.DataFrame()
    rows = []
    for r in gsc_queries:
        keys = r.get("keys", [])
        query = keys[0] if keys else ""
        pos = r.get("position", 0)
        impr = r.get("impressions", 0)
        clicks = r.get("clicks", 0)
        ctr = r.get("ctr", 0)
        if not (4 <= pos <= 20 and impr > 100):
            continue
        expected = EXPECTED_CTR.get(min(int(round(pos)), 20), 0.01)
        if ctr >= expected:
            continue
        potential_ctr = EXPECTED_CTR.get(3, 0.11)
        potential_clicks = int(impr * potential_ctr)
        rows.append({
            "Requête": query,
            "Position": round(pos, 1),
            "Impressions": impr,
            "Clicks": clicks,
            "CTR actuel": f"{ctr:.1%}",
            "CTR attendu": f"{expected:.1%}",
            "CTR potentiel (top 3)": f"{potential_ctr:.1%}",
            "Gain estimé clicks": potential_clicks - clicks,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Gain estimé clicks", ascending=False)
    return df


def calculate_content_decay(current_pages: Optional[List[dict]],
                            prev_pages: Optional[List[dict]]) -> pd.DataFrame:
    """Pages dont les clicks ont baissé de > 20 % vs période précédente."""
    if not current_pages or not prev_pages:
        return pd.DataFrame()
    cur = {r["keys"][0]: r.get("clicks", 0) for r in current_pages if r.get("keys")}
    prev = {r["keys"][0]: r.get("clicks", 0) for r in prev_pages if r.get("keys")}
    rows = []
    for page, prev_c in prev.items():
        if prev_c < 10:
            continue
        cur_c = cur.get(page, 0)
        delta = ((cur_c - prev_c) / prev_c) * 100
        if delta < -20:
            short = page.split("//", 1)[-1] if "//" in page else page
            rows.append({
                "URL": short,
                "Clicks N": cur_c,
                "Clicks N-1": prev_c,
                "Delta %": f"{delta:+.1f}%",
                "_delta": delta,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_delta").drop(columns=["_delta"])
    return df


def calculate_cannibalization(query_page_data: Optional[List[dict]]) -> pd.DataFrame:
    """Requêtes GSC rankant avec 2+ URLs."""
    if not query_page_data:
        return pd.DataFrame()
    qmap: dict = defaultdict(list)
    for r in query_page_data:
        keys = r.get("keys", [])
        if len(keys) >= 2:
            qmap[keys[0]].append({
                "url": keys[1],
                "position": round(r.get("position", 0), 1),
                "clicks": r.get("clicks", 0),
            })
    rows = []
    for q, urls in qmap.items():
        if len(urls) < 2:
            continue
        urls_s = sorted(urls, key=lambda u: u["position"])
        detail = "; ".join(
            f"{u['url'].split('//', 1)[-1]} (pos {u['position']})"
            for u in urls_s[:3]
        )
        rows.append({
            "Requête": q,
            "Nb URLs": len(urls),
            "URLs & Positions": detail,
            "Clicks total": sum(u["clicks"] for u in urls),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Clicks total", ascending=False)
    return df


# ============================================================
# Helpers pour extraire les rows des réponses Ahrefs
# ============================================================

def _rows(data: Optional[dict]) -> list:
    """Extrait la liste de lignes d'une réponse Ahrefs."""
    return AhrefsAPI._extract_rows(data)


# ============================================================
# Affichage : KPIs
# ============================================================

def display_kpis(data: dict):
    ahrefs = data.get("ahrefs", {})
    dr_data = ahrefs.get("domain_rating") or {}
    metrics = ahrefs.get("metrics") or {}
    bl = ahrefs.get("backlinks_stats") or {}
    gsc = data.get("gsc", {})

    # Deltas GSC
    cur_q = gsc.get("current_period_queries") or []
    prev_q = gsc.get("prev_period_queries") or []
    cur_clicks = sum(r.get("clicks", 0) for r in cur_q)
    prev_clicks = sum(r.get("clicks", 0) for r in prev_q)
    cur_impr = sum(r.get("impressions", 0) for r in cur_q)
    prev_impr = sum(r.get("impressions", 0) for r in prev_q)

    cols = st.columns(6)
    with cols[0]:
        st.metric("Domain Rating", dr_data.get("domain_rating", "N/A"))
    with cols[1]:
        st.metric("Trafic organique", f"{metrics.get('org_traffic', 0):,}")
    with cols[2]:
        st.metric("Keywords top 100", f"{metrics.get('org_keywords', 0):,}")
    with cols[3]:
        st.metric("Referring Domains", f"{bl.get('live_refdomains', 0):,}" if bl else "N/A")
    with cols[4]:
        delta_c = cur_clicks - prev_clicks if prev_clicks else None
        st.metric("Clicks GSC (28j)", f"{cur_clicks:,}",
                  delta=f"{delta_c:+,}" if delta_c is not None else None)
    with cols[5]:
        delta_i = cur_impr - prev_impr if prev_impr else None
        st.metric("Impressions GSC (28j)", f"{cur_impr:,}",
                  delta=f"{delta_i:+,}" if delta_i is not None else None)


# ============================================================
# Affichage : Graphique d'évolution
# ============================================================

def display_traffic_chart(data: dict):
    hist = data.get("ahrefs", {}).get("metrics_history")
    rows = _rows(hist)
    if not rows:
        st.info("Données historiques Ahrefs indisponibles.")
        return
    df = pd.DataFrame(rows)
    if "date" not in df.columns or "org_traffic" not in df.columns:
        st.warning("Format de données historiques inattendu.")
        return
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["org_traffic"],
        mode="lines+markers", name="Trafic organique (Ahrefs)",
        line=dict(color="#667eea", width=3), marker=dict(size=6),
    ))
    if "paid_traffic" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["paid_traffic"],
            mode="lines+markers", name="Trafic payant",
            line=dict(color="#F59E0B", width=2, dash="dot"), marker=dict(size=4),
        ))
    fig.update_layout(
        title="Évolution du trafic sur 12 mois",
        xaxis_title="Date", yaxis_title="Visiteurs estimés / mois",
        template="plotly_white", hovermode="x unified", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Affichage : Comparaison concurrents
# ============================================================

def display_competitors_comparison(data: dict):
    domain = data.get("domain", "")
    ahrefs = data.get("ahrefs", {})
    dr = ahrefs.get("domain_rating") or {}
    met = ahrefs.get("metrics") or {}
    bl = ahrefs.get("backlinks_stats") or {}

    rows = [{
        "Domaine": f"{domain}",
        "DR": dr.get("domain_rating", "N/A"),
        "Trafic organique": met.get("org_traffic", 0),
        "Keywords": met.get("org_keywords", 0),
        "Ref. Domains": bl.get("live_refdomains", 0) if bl else 0,
        "Backlinks": bl.get("live", 0) if bl else 0,
    }]

    for comp, cd in ahrefs.get("competitors_data", {}).items():
        c_dr = cd.get("domain_rating") or {}
        c_m = cd.get("metrics") or {}
        c_bl = cd.get("backlinks_stats") or {}
        rows.append({
            "Domaine": comp,
            "DR": c_dr.get("domain_rating", "N/A"),
            "Trafic organique": c_m.get("org_traffic", 0),
            "Keywords": c_m.get("org_keywords", 0),
            "Ref. Domains": c_bl.get("live_refdomains", 0) if c_bl else 0,
            "Backlinks": c_bl.get("live", 0) if c_bl else 0,
        })

    if len(rows) > 1:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun concurrent renseigné pour la comparaison.")


# ============================================================
# Affichage : Dashboard complet
# ============================================================

def display_dashboard(data: dict, qw: pd.DataFrame, cd: pd.DataFrame, cn: pd.DataFrame):
    st.markdown('<p class="section-title">Dashboard de données</p>', unsafe_allow_html=True)

    display_kpis(data)
    st.divider()

    tabs = st.tabs([
        "📈 Évolution", "🏆 Quick Wins", "📉 Content Decay",
        "⚠️ Cannibalization", "🏅 Concurrents", "🔗 Backlinks",
        "🌍 Pays", "📱 Devices",
    ])

    # ---- Évolution ----
    with tabs[0]:
        display_traffic_chart(data)
        top_pages_raw = data.get("ahrefs", {}).get("top_pages")
        tp_rows = _rows(top_pages_raw)
        if tp_rows:
            st.subheader("Top 20 pages par trafic organique (Ahrefs)")
            dfp = pd.DataFrame(tp_rows)
            rn = {
                "url": "URL", "sum_traffic": "Trafic", "top_keyword": "Top Keyword",
                "top_keyword_best_position": "Position", "keywords": "Nb Keywords",
                "value": "Valeur (cents $)",
            }
            dfp = dfp.rename(columns={k: v for k, v in rn.items() if k in dfp.columns})
            st.dataframe(dfp, use_container_width=True, hide_index=True)

        # Top organic keywords
        ok_raw = data.get("ahrefs", {}).get("organic_keywords")
        ok_rows = _rows(ok_raw)
        if ok_rows:
            st.subheader("Top 50 keywords organiques (Ahrefs)")
            dfk = pd.DataFrame(ok_rows)
            rn2 = {
                "keyword": "Keyword", "volume": "Volume", "best_position": "Position",
                "best_position_url": "URL", "sum_traffic": "Trafic estimé",
                "keyword_difficulty": "KD", "cpc": "CPC (cents $)",
            }
            dfk = dfk.rename(columns={k: v for k, v in rn2.items() if k in dfk.columns})
            st.dataframe(dfk, use_container_width=True, hide_index=True)

    # ---- Quick Wins ----
    with tabs[1]:
        if not qw.empty:
            st.markdown(
                f'<div class="success-box">🎯 <strong>{len(qw)} quick wins identifiés</strong>'
                f' — requêtes en position 4-20 avec CTR sous-optimal</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(qw, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun quick win identifié (données GSC nécessaires).")

    # ---- Content Decay ----
    with tabs[2]:
        if not cd.empty:
            st.markdown(
                f'<div class="warning-box">📉 <strong>{len(cd)} pages en déclin</strong>'
                f' — baisse de clicks > 20% vs période précédente</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(cd, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune page en déclin détectée (données GSC nécessaires).")

    # ---- Cannibalization ----
    with tabs[3]:
        if not cn.empty:
            st.markdown(
                f'<div class="critical-box">⚠️ <strong>{len(cn)} cas de cannibalization</strong>'
                f' — requêtes faisant ranker 2+ URLs</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(cn, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune cannibalization détectée (données GSC nécessaires).")

    # ---- Concurrents ----
    with tabs[4]:
        st.subheader("Comparaison vs concurrents")
        display_competitors_comparison(data)

        oc_raw = data.get("ahrefs", {}).get("organic_competitors")
        oc_rows = _rows(oc_raw)
        if oc_rows:
            st.subheader("Concurrents organiques détectés (Ahrefs)")
            dfc = pd.DataFrame(oc_rows)
            rn3 = {
                "competitor_domain": "Domaine", "keywords_common": "KW communs",
                "keywords_competitor": "KW concurrent", "keywords_target": "KW vous",
                "share": "Part commune", "traffic": "Trafic", "domain_rating": "DR",
            }
            dfc = dfc.rename(columns={k: v for k, v in rn3.items() if k in dfc.columns})
            if "Part commune" in dfc.columns:
                dfc["Part commune"] = dfc["Part commune"].apply(
                    lambda x: f"{x * 100:.1f}%" if isinstance(x, (int, float)) else x
                )
            st.dataframe(dfc, use_container_width=True, hide_index=True)

    # ---- Backlinks ----
    with tabs[5]:
        rd_raw = data.get("ahrefs", {}).get("referring_domains")
        rd_rows = _rows(rd_raw)
        if rd_rows:
            st.subheader("Top referring domains (par DR)")
            dfr = pd.DataFrame(rd_rows)
            rn4 = {
                "domain": "Domaine", "domain_rating": "DR",
                "dofollow_links": "Liens dofollow", "links_to_target": "Liens total",
                "first_seen": "Découvert le", "traffic_domain": "Trafic domaine",
            }
            dfr = dfr.rename(columns={k: v for k, v in rn4.items() if k in dfr.columns})
            st.dataframe(dfr, use_container_width=True, hide_index=True)
        else:
            st.info("Données referring domains indisponibles.")

    # ---- Pays ----
    with tabs[6]:
        bc_raw = data.get("ahrefs", {}).get("metrics_by_country")
        bc_rows = _rows(bc_raw)
        if bc_rows:
            st.subheader("Répartition du trafic organique par pays")
            dfco = pd.DataFrame(bc_rows)
            if "org_traffic" in dfco.columns:
                dfco = dfco[dfco["org_traffic"] > 0].sort_values("org_traffic", ascending=False)
            rn5 = {
                "country": "Pays", "org_traffic": "Trafic organique",
                "org_keywords": "Keywords", "org_keywords_1_3": "KW top 3",
                "org_cost": "Valeur (cents $)",
            }
            dfco = dfco.rename(columns={k: v for k, v in rn5.items() if k in dfco.columns})
            if not dfco.empty and "Trafic organique" in dfco.columns:
                fig = px.pie(
                    dfco.head(10), values="Trafic organique", names="Pays",
                    title="Top 10 pays par trafic organique",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dfco.head(15), use_container_width=True, hide_index=True)
        else:
            st.info("Données de répartition par pays indisponibles.")

    # ---- Devices ----
    with tabs[7]:
        devs = data.get("gsc", {}).get("devices")
        if devs:
            st.subheader("Répartition par type d'appareil (GSC)")
            dfd = pd.DataFrame([{
                "Device": r.get("keys", [""])[0],
                "Clicks": r.get("clicks", 0),
                "Impressions": r.get("impressions", 0),
                "CTR": f"{r.get('ctr', 0):.1%}",
                "Position moy.": round(r.get("position", 0), 1),
            } for r in devs])
            fig = px.pie(dfd, values="Clicks", names="Device",
                         title="Répartition des clicks par device")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dfd, use_container_width=True, hide_index=True)
        else:
            st.info("Données devices GSC indisponibles.")

    # ---- Erreurs ----
    errors = data.get("errors", [])
    if errors:
        with st.expander(f"⚠️ {len(errors)} source(s) de données indisponible(s)", expanded=False):
            for e in errors:
                st.warning(e)


# ============================================================
# Analyse stratégique IA (Claude)
# ============================================================

SYSTEM_PROMPT_ANALYSIS = """Tu es un consultant SEO senior avec 15 ans d'expérience. On te fournit les données SEO complètes d'un site client. Tu dois produire un diagnostic stratégique structuré et actionnable.

Analyse les données et produis :

1. **Résumé exécutif** (3-4 phrases) : état de santé SEO du site, tendance générale, position vs concurrents

2. **Forces identifiées** (2-3 points) : ce qui fonctionne bien, les assets du site

3. **Faiblesses et problèmes** (3-5 points) : les problèmes critiques détectés dans les données, avec preuves chiffrées

4. **Opportunités prioritaires** (5 actions maximum) : plan d'action concret, classé par impact estimé (fort/moyen) et effort (faible/moyen/élevé). Chaque action doit être spécifique et inclure :
   - Ce qu'il faut faire concrètement
   - Pourquoi (données à l'appui)
   - Impact attendu estimé
   - Effort/complexité

5. **Quick wins immédiats** : les 3 actions à faire cette semaine basées sur les données quick wins et content decay

Adapte ton analyse aux objectifs et contraintes du client fournis. Sois concret, évite les généralités. Chaque recommandation doit être directement liée à une donnée du diagnostic."""


def _build_analysis_payload(data: dict, objectives: List[str], constraints: List[str],
                            qw: pd.DataFrame, cd: pd.DataFrame, cn: pd.DataFrame) -> str:
    """Construit le message user pour l'analyse Claude."""
    payload: dict = {
        "domaine": data.get("domain", ""),
        "objectifs": objectives,
        "contraintes": constraints,
        "ahrefs": {},
        "gsc": {},
        "analyses_derivees": {},
    }

    ah = data.get("ahrefs", {})
    payload["ahrefs"]["domain_rating"] = ah.get("domain_rating")
    payload["ahrefs"]["metrics"] = ah.get("metrics")
    payload["ahrefs"]["backlinks_stats"] = ah.get("backlinks_stats")

    tp = _rows(ah.get("top_pages"))
    if tp:
        payload["ahrefs"]["top_pages"] = tp[:10]

    ok = _rows(ah.get("organic_keywords"))
    if ok:
        payload["ahrefs"]["top_keywords"] = ok[:20]

    hist = _rows(ah.get("metrics_history"))
    if hist:
        payload["ahrefs"]["metrics_history_12m"] = hist

    payload["ahrefs"]["competitors_data"] = ah.get("competitors_data", {})

    oc = _rows(ah.get("organic_competitors"))
    if oc:
        payload["ahrefs"]["organic_competitors"] = oc[:5]

    rd = _rows(ah.get("referring_domains"))
    if rd:
        payload["ahrefs"]["top_referring_domains"] = rd[:10]

    gsc = data.get("gsc", {})
    payload["gsc"]["performance"] = gsc.get("performance")

    tq = gsc.get("top_queries")
    if tq:
        payload["gsc"]["top_queries"] = [{
            "query": r["keys"][0], "clicks": r.get("clicks", 0),
            "impressions": r.get("impressions", 0),
            "ctr": round(r.get("ctr", 0), 4),
            "position": round(r.get("position", 0), 1),
        } for r in tq[:20]]

    devs = gsc.get("devices")
    if devs:
        payload["gsc"]["devices"] = [{
            "device": r["keys"][0], "clicks": r.get("clicks", 0),
            "ctr": round(r.get("ctr", 0), 4),
        } for r in devs]

    if not qw.empty:
        payload["analyses_derivees"]["quick_wins"] = qw.head(15).to_dict(orient="records")
    if not cd.empty:
        payload["analyses_derivees"]["content_decay"] = cd.head(15).to_dict(orient="records")
    if not cn.empty:
        payload["analyses_derivees"]["cannibalization"] = cn.head(10).to_dict(orient="records")

    return json.dumps(payload, indent=2, default=str, ensure_ascii=False)


def generate_strategic_analysis(data: dict, objectives: List[str], constraints: List[str],
                                qw: pd.DataFrame, cd: pd.DataFrame, cn: pd.DataFrame,
                                api_key: str) -> Optional[str]:
    """Appelle Claude pour produire l'analyse stratégique."""
    user_msg = (
        f"Voici les données SEO complètes du site {data.get('domain', '')} :\n\n"
        + _build_analysis_payload(data, objectives, constraints, qw, cd, cn)
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT_ANALYSIS,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return None


# ============================================================
# Export Markdown
# ============================================================

def generate_export_md(data: dict, analysis: str,
                       qw: pd.DataFrame, cd: pd.DataFrame, cn: pd.DataFrame) -> str:
    domain = data.get("domain", "")
    ah = data.get("ahrefs", {})
    dr = ah.get("domain_rating") or {}
    met = ah.get("metrics") or {}
    bl = ah.get("backlinks_stats") or {}

    md = f"""# Diagnostic SEO — {domain}
*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*

---

## KPIs Principaux

| Métrique | Valeur |
|----------|--------|
| Domain Rating | {dr.get('domain_rating', 'N/A')} |
| Trafic organique | {met.get('org_traffic', 0):,} |
| Keywords totaux | {met.get('org_keywords', 0):,} |
| Referring Domains | {bl.get('live_refdomains', 0) if bl else 0:,} |
| Backlinks | {bl.get('live', 0) if bl else 0:,} |

"""

    if not qw.empty:
        md += "## Quick Wins\n\n"
        md += qw.to_markdown(index=False) + "\n\n"

    if not cd.empty:
        md += "## Content Decay\n\n"
        md += cd.to_markdown(index=False) + "\n\n"

    if not cn.empty:
        md += "## Cannibalization\n\n"
        md += cn.to_markdown(index=False) + "\n\n"

    # Comparaison concurrents
    comp = ah.get("competitors_data", {})
    if comp:
        md += "## Comparaison Concurrents\n\n"
        md += "| Domaine | DR | Trafic | Keywords | Ref Domains |\n"
        md += "|---------|---:|-------:|--------:|------------:|\n"
        md += (f"| **{domain}** | {dr.get('domain_rating', 'N/A')} "
               f"| {met.get('org_traffic', 0):,} | {met.get('org_keywords', 0):,} "
               f"| {bl.get('live_refdomains', 0) if bl else 0:,} |\n")
        for c, cd_ in comp.items():
            c_dr = cd_.get("domain_rating") or {}
            c_m = cd_.get("metrics") or {}
            c_bl = cd_.get("backlinks_stats") or {}
            md += (f"| {c} | {c_dr.get('domain_rating', 'N/A')} "
                   f"| {c_m.get('org_traffic', 0):,} | {c_m.get('org_keywords', 0):,} "
                   f"| {c_bl.get('live_refdomains', 0) if c_bl else 0:,} |\n")
        md += "\n"

    if analysis:
        md += "---\n\n## Analyse Stratégique IA\n\n" + analysis + "\n"

    md += "\n---\n*Rapport généré par SEO Strategy Advisor*\n"
    return md


# ============================================================
# Main
# ============================================================

def main():
    st.set_page_config(
        page_title="Diagnostic SEO Client",
        page_icon="🎯",
        layout="wide",
    )

    check_password()
    inject_css()

    st.markdown('<p class="main-header">🎯 Diagnostic SEO Client</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Diagnostic stratégique complet en quelques minutes '
        '— Ahrefs + GSC + IA</p>',
        unsafe_allow_html=True,
    )

    # ---- Sidebar : clés API ----
    with st.sidebar:
        st.header("⚙️ Configuration API")

        st.subheader("🔗 Ahrefs")
        ahrefs_token = st.text_input(
            "Token API Ahrefs",
            value=st.secrets.get("AHREFS_API_TOKEN", ""),
            type="password",
        )

        st.subheader("🤖 Claude (Anthropic)")
        anthropic_key = st.text_input(
            "Clé API Anthropic",
            value=st.secrets.get("ANTHROPIC_API_KEY", ""),
            type="password",
        )

        st.divider()
        st.subheader("🔍 Google Search Console")
        gsc = GSCAPI()
        if gsc.is_configured:
            st.success("GSC connecté")
        else:
            st.warning(
                "GSC non configuré — Ajoutez GSC_CLIENT_ID, GSC_CLIENT_SECRET "
                "et GSC_REFRESH_TOKEN dans vos secrets Streamlit."
            )

    # ---- Session state ----
    if "diagnostic_data" not in st.session_state:
        st.session_state.diagnostic_data = None
    if "strategic_analysis" not in st.session_state:
        st.session_state.strategic_analysis = None

    # ===== ÉTAPE 1 : Configuration client =====
    st.markdown('<p class="section-title">Configuration du client</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        domain = st.text_input(
            "Domaine du client", placeholder="example.com",
            help="Sans protocole (ex : example.com)",
        )

        st.markdown("**Concurrents** (optionnel, jusqu'à 3)")
        comp1 = st.text_input("Concurrent 1", placeholder="concurrent1.com", key="comp1")
        comp2 = st.text_input("Concurrent 2", placeholder="concurrent2.com", key="comp2")
        comp3 = st.text_input("Concurrent 3", placeholder="concurrent3.com", key="comp3")
        competitors = [c.strip() for c in [comp1, comp2, comp3] if c.strip()]

    with col2:
        # Propriété GSC
        if gsc.is_configured:
            props = gsc.list_properties()
            if props:
                gsc_property = st.selectbox(
                    "Propriété Google Search Console",
                    ["(Aucune — sans GSC)"] + props,
                )
                if gsc_property.startswith("("):
                    gsc_property = ""
            else:
                gsc_property = st.text_input(
                    "Propriété GSC (format exact)",
                    placeholder="https://example.com/ ou sc-domain:example.com",
                )
        else:
            gsc_property = ""
            st.text_input(
                "Propriété GSC (format exact)",
                placeholder="Configurez GSC dans les secrets",
                disabled=True,
            )

        objectives = st.multiselect(
            "Objectifs du client", OBJECTIVES_OPTIONS,
            default=["Augmenter le trafic organique"],
        )

        constraints = st.multiselect("Contraintes (optionnel)", CONSTRAINTS_OPTIONS)

        country_code = st.selectbox(
            "Pays principal",
            options=[c[0] for c in COUNTRY_OPTIONS],
            format_func=lambda x: next(f"{c[0]} — {c[1]}" for c in COUNTRY_OPTIONS if c[0] == x),
            index=0,
        )

    # Validation
    can_launch = bool(domain.strip() and ahrefs_token.strip())
    if not ahrefs_token.strip():
        st.info("Renseignez votre token API Ahrefs dans la sidebar pour commencer.")

    # ===== ÉTAPE 2 : Collecte =====
    if st.button("🚀 Lancer le diagnostic", type="primary",
                 disabled=not can_launch, use_container_width=True):
        data = collect_all_data(
            domain=domain.strip(),
            competitors=competitors,
            gsc_property=gsc_property.strip() if gsc_property else "",
            ahrefs_token=ahrefs_token.strip(),
            country=country_code,
        )
        st.session_state.diagnostic_data = data
        st.session_state.strategic_analysis = None

    # ===== ÉTAPES 3-5 : Dashboard + Analyse + Export =====
    if st.session_state.diagnostic_data:
        data = st.session_state.diagnostic_data

        # Calculs dérivés
        qw = calculate_quick_wins(data.get("gsc", {}).get("top_queries"))
        cd_df = calculate_content_decay(
            data.get("gsc", {}).get("current_period_pages"),
            data.get("gsc", {}).get("prev_period_pages"),
        )
        cn_df = calculate_cannibalization(data.get("gsc", {}).get("query_page"))

        # Dashboard
        display_dashboard(data, qw, cd_df, cn_df)

        # ---- Analyse IA ----
        st.divider()
        st.markdown('<p class="section-title">Analyse stratégique IA</p>', unsafe_allow_html=True)

        if not anthropic_key.strip():
            st.warning("Renseignez votre clé API Anthropic dans la sidebar pour générer l'analyse.")
        else:
            if st.button("🧠 Générer l'analyse stratégique",
                         type="secondary", use_container_width=True):
                with st.spinner("Analyse en cours par Claude (claude-sonnet-4-20250514)..."):
                    result = generate_strategic_analysis(
                        data, objectives, constraints, qw, cd_df, cn_df,
                        anthropic_key.strip(),
                    )
                if result:
                    st.session_state.strategic_analysis = result
                else:
                    st.error("Erreur lors de l'analyse. Vérifiez votre clé API Anthropic.")

        # ---- Affichage analyse ----
        if st.session_state.strategic_analysis:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown(st.session_state.strategic_analysis)
            st.markdown('</div>', unsafe_allow_html=True)

            # ---- Export ----
            st.divider()
            export_md = generate_export_md(
                data, st.session_state.strategic_analysis, qw, cd_df, cn_df,
            )
            fname = f"diagnostic_seo_{data.get('domain', 'site')}_{datetime.now().strftime('%Y%m%d')}.md"
            st.download_button(
                "📥 Exporter le rapport complet (Markdown)",
                data=export_md,
                file_name=fname,
                mime="text/markdown",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
