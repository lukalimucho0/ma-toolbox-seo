"""
📊 RAPPORT CLIENT AUTOMATISÉ
==============================
Génère un rapport SEO mensuel professionnel (.docx) pour chaque client.
Combine GSC API + Ahrefs API v3 + DataForSEO + Claude pour produire
un livrable complet : KPIs, évolutions, top pages, recommandations.
"""

import streamlit as st
import requests
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import anthropic
import json
import re
import io
import os
import time
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT

from utils.auth import check_password

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Rapport Client | Ma Toolbox SEO",
    page_icon="📊",
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
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A5F;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .kpi-up { color: #22C55E; font-weight: bold; }
    .kpi-down { color: #EF4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CLIENTS — Chargement des configs JSON
# =============================================================================
CLIENTS_DIR = os.path.join(os.path.dirname(__file__), "clients")


def load_clients() -> dict:
    clients = {}
    if os.path.isdir(CLIENTS_DIR):
        for fname in sorted(os.listdir(CLIENTS_DIR)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(CLIENTS_DIR, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    name = data.get("name", fname.replace(".json", ""))
                    domain = data.get("domain", "")
                    if domain:
                        clients[name] = data
                except Exception:
                    pass
    return clients


# =============================================================================
# API CLIENTS
# =============================================================================

AHREFS_API_BASE = "https://api.ahrefs.com/v3"


class AhrefsAPI:
    def __init__(self, api_token: str):
        self.base_url = AHREFS_API_BASE
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        })

    def _get(self, endpoint: str, params: dict):
        url = f"{self.base_url}/{endpoint}"
        params["output"] = "json"
        try:
            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Ahrefs API error [{endpoint}]: {e}")
            return None

    @staticmethod
    def _rows(data):
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
        return []

    def get_domain_rating(self, target: str, date: str):
        return self._get("site-explorer/domain-rating", {"target": target, "date": date})

    def get_metrics(self, target: str, date: str, mode: str = "subdomains"):
        return self._get("site-explorer/metrics", {"target": target, "date": date, "mode": mode})

    def get_backlinks_stats(self, target: str, date: str, mode: str = "subdomains"):
        return self._get("site-explorer/backlinks-stats", {"target": target, "date": date, "mode": mode})

    def get_metrics_history(self, target: str, date_from: str, date_to: str, mode: str = "subdomains"):
        return self._get("site-explorer/metrics-history", {
            "target": target, "date_from": date_from, "date_to": date_to,
            "mode": mode, "history_grouping": "monthly",
            "select": "date,org_traffic,org_cost,org_keywords,paid_traffic",
        })

    def get_top_pages(self, target: str, date: str, country: str = None, limit: int = 20, mode: str = "subdomains"):
        params = {
            "target": target, "date": date, "mode": mode,
            "select": "url,sum_traffic,top_keyword,top_keyword_best_position,keywords,value",
            "order_by": "sum_traffic:desc", "limit": limit,
        }
        if country:
            params["country"] = country
        return self._get("site-explorer/top-pages", params)

    def get_organic_keywords(self, target: str, date: str, country: str = "FR",
                             limit: int = 30, mode: str = "subdomains"):
        return self._get("site-explorer/organic-keywords", {
            "target": target, "date": date, "mode": mode,
            "country": country, "limit": limit,
            "select": "keyword,volume,best_position,best_position_url,sum_traffic,keyword_difficulty,best_position_diff",
            "order_by": "sum_traffic:desc",
        })

    def get_refdomains_history(self, target: str, date_from: str, date_to: str, mode: str = "subdomains"):
        return self._get("site-explorer/refdomains-history", {
            "target": target, "date_from": date_from, "date_to": date_to,
            "mode": mode, "history_grouping": "monthly",
        })


class DataForSEOLabs:
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
                            limit: int = 50) -> dict:
        return self._post("/dataforseo_labs/google/ranked_keywords/live", [{
            "target": domain,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
            "item_types": ["organic"],
            "order_by": ["keyword_data.keyword_info.search_volume,desc"]
        }])


class GSCAPI:
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
                    token=None, refresh_token=refresh_token,
                    client_id=client_id, client_secret=client_secret,
                    token_uri="https://oauth2.googleapis.com/token",
                )
                self.service = build("searchconsole", "v1", credentials=creds)
                return
            if "GSC_SERVICE_ACCOUNT" in st.secrets:
                from google.oauth2 import service_account
                sa_info = dict(st.secrets["GSC_SERVICE_ACCOUNT"])
                creds = service_account.Credentials.from_service_account_info(
                    sa_info,
                    scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
                )
                self.service = build("searchconsole", "v1", credentials=creds)
        except Exception as e:
            logger.warning(f"GSC init: {e}")

    @property
    def is_configured(self) -> bool:
        return self.service is not None

    def search_analytics(self, site_url: str, dimensions: list,
                         start_date: str, end_date: str, row_limit: int = 1000):
        if not self.service:
            return None
        try:
            body = {
                "startDate": start_date, "endDate": end_date,
                "dimensions": dimensions, "rowLimit": row_limit,
                "dataState": "final",
            }
            resp = self.service.searchanalytics().query(siteUrl=site_url, body=body).execute()
            return resp.get("rows", [])
        except Exception as e:
            logger.error(f"GSC error: {e}")
            return None

    def performance_totals(self, site_url: str, start_date: str, end_date: str):
        if not self.service:
            return None
        try:
            body = {"startDate": start_date, "endDate": end_date, "dataState": "final"}
            resp = self.service.searchanalytics().query(siteUrl=site_url, body=body).execute()
            rows = resp.get("rows", [])
            return rows[0] if rows else None
        except Exception as e:
            logger.error(f"GSC totals error: {e}")
            return None


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
    return domain.rstrip('/')


def format_number(n) -> str:
    if n is None:
        return "N/A"
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def delta_str(current, previous):
    if previous is None or previous == 0:
        return "N/A", "neutral"
    diff = current - previous
    pct = (diff / abs(previous)) * 100
    sign = "+" if diff >= 0 else ""
    direction = "up" if diff > 0 else "down" if diff < 0 else "neutral"
    return f"{sign}{pct:.1f}%", direction


def generate_recommendations(api_key: str, report_data: dict) -> str:
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Tu es un consultant SEO senior. Rédige les recommandations du rapport mensuel pour le client "{report_data['client_name']}" (domaine : {report_data['domain']}).

## Données du mois :
- Trafic organique Ahrefs : {report_data.get('current_traffic', 'N/A')} (évolution : {report_data.get('traffic_delta', 'N/A')})
- Mots-clés organiques : {report_data.get('current_keywords', 'N/A')} (évolution : {report_data.get('keywords_delta', 'N/A')})
- Domain Rating : {report_data.get('domain_rating', 'N/A')}
- Domaines référents : {report_data.get('refdomains', 'N/A')}

## Top pages (par trafic) :
{report_data.get('top_pages_summary', 'Non disponible')}

## Mots-clés principaux (mouvements) :
{report_data.get('keywords_summary', 'Non disponible')}

## GSC (si disponible) :
- Clics : {report_data.get('gsc_clicks', 'N/A')} | Impressions : {report_data.get('gsc_impressions', 'N/A')}
- CTR moyen : {report_data.get('gsc_ctr', 'N/A')} | Position moyenne : {report_data.get('gsc_position', 'N/A')}

## Rédige :
1. **Synthèse du mois** (3-4 phrases) : ce qui s'est passé, la tendance générale
2. **Points positifs** (2-3 bullets) : ce qui a bien fonctionné
3. **Points d'attention** (2-3 bullets) : ce qui nécessite une action
4. **Actions recommandées pour le mois prochain** (3-5 bullets priorisées) : actions concrètes et spécifiques

Sois concis, factuel, orienté action. Pas de blabla générique.
Réponds en texte structuré (pas de JSON), avec des titres en markdown.
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# =============================================================================
# GÉNÉRATION DU DOCUMENT .DOCX
# =============================================================================

def set_cell_shading(cell, color_hex: str):
    from docx.oxml.ns import qn
    from lxml import etree
    shading = etree.SubElement(cell._tc.get_or_add_tcPr(), qn("w:shd"))
    shading.set(qn("w:fill"), color_hex)
    shading.set(qn("w:val"), "clear")


def add_styled_table(doc, headers: list, rows: list, col_widths: list = None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)
        set_cell_shading(cell, "667eea")
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.color.rgb = RGBColor(255, 255, 255)

    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    return table


def create_report_docx(report: dict) -> io.BytesIO:
    doc = Document()

    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(10)

    # --- Page de garde ---
    doc.add_paragraph("")
    doc.add_paragraph("")
    title = doc.add_heading(f"Rapport SEO Mensuel", level=0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    subtitle = doc.add_heading(report["client_name"], level=1)
    subtitle.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    for run in subtitle.runs:
        run.font.color.rgb = RGBColor(102, 126, 234)

    period = doc.add_paragraph(f"Période : {report['period']}")
    period.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    date_gen = doc.add_paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y')}")
    date_gen.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    for run in date_gen.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(150, 150, 150)

    doc.add_page_break()

    # --- Table des matières (manuelle) ---
    doc.add_heading("Sommaire", level=1)
    toc_items = [
        "1. KPIs du mois",
        "2. Évolution du trafic organique",
        "3. Top pages par trafic",
        "4. Mots-clés principaux",
        "5. Profil de liens",
        "6. Google Search Console",
        "7. Synthèse & Recommandations",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.style = "List Number"

    doc.add_page_break()

    # --- 1. KPIs ---
    doc.add_heading("1. KPIs du mois", level=1)

    kpis = report.get("kpis", {})
    kpi_headers = ["Indicateur", "Valeur actuelle", "Mois précédent", "Évolution"]
    kpi_rows = []
    for label, data in kpis.items():
        kpi_rows.append([
            label,
            str(data.get("current", "N/A")),
            str(data.get("previous", "N/A")),
            data.get("delta", "N/A"),
        ])
    if kpi_rows:
        add_styled_table(doc, kpi_headers, kpi_rows, [5, 3.5, 3.5, 3])

    doc.add_paragraph("")

    # --- 2. Évolution trafic ---
    doc.add_heading("2. Évolution du trafic organique", level=1)

    history = report.get("traffic_history", [])
    if history:
        p = doc.add_paragraph("Évolution mensuelle du trafic organique estimé (Ahrefs) :")
        p.italic = True

        hist_headers = ["Mois", "Trafic organique", "Mots-clés", "Valeur ($)"]
        hist_rows = [[h["date"], format_number(h.get("traffic", 0)),
                       format_number(h.get("keywords", 0)),
                       f"${format_number(h.get('cost', 0))}"] for h in history]
        add_styled_table(doc, hist_headers, hist_rows, [4, 3.5, 3.5, 3.5])
    else:
        doc.add_paragraph("Données non disponibles.")

    doc.add_paragraph("")

    # --- 3. Top pages ---
    doc.add_heading("3. Top pages par trafic", level=1)

    top_pages = report.get("top_pages", [])
    if top_pages:
        tp_headers = ["URL", "Trafic", "Top mot-clé", "Position"]
        tp_rows = []
        for pg in top_pages[:15]:
            url_short = pg.get("url", "")
            if len(url_short) > 60:
                url_short = url_short[:57] + "..."
            tp_rows.append([
                url_short,
                format_number(pg.get("traffic", 0)),
                pg.get("top_keyword", ""),
                str(pg.get("position", "")),
            ])
        add_styled_table(doc, tp_headers, tp_rows, [7, 2.5, 3.5, 2])
    else:
        doc.add_paragraph("Données non disponibles.")

    doc.add_paragraph("")

    # --- 4. Mots-clés ---
    doc.add_heading("4. Mots-clés principaux", level=1)

    keywords = report.get("keywords", [])
    if keywords:
        kw_headers = ["Mot-clé", "Volume", "Position", "Mouvement", "Trafic"]
        kw_rows = []
        for kw in keywords[:25]:
            diff = kw.get("position_diff", 0) or 0
            if diff > 0:
                mvt = f"↑ +{diff}"
            elif diff < 0:
                mvt = f"↓ {diff}"
            else:
                mvt = "="
            kw_rows.append([
                kw.get("keyword", ""),
                format_number(kw.get("volume", 0)),
                str(kw.get("position", "")),
                mvt,
                format_number(kw.get("traffic", 0)),
            ])
        add_styled_table(doc, kw_headers, kw_rows, [5, 2, 2, 2.5, 2.5])
    else:
        doc.add_paragraph("Données non disponibles.")

    doc.add_paragraph("")

    # --- 5. Profil de liens ---
    doc.add_heading("5. Profil de liens", level=1)

    backlinks = report.get("backlinks", {})
    if backlinks:
        bl_headers = ["Métrique", "Valeur"]
        bl_rows = [
            ["Domain Rating", str(backlinks.get("domain_rating", "N/A"))],
            ["Backlinks totaux", format_number(backlinks.get("live_backlinks", 0))],
            ["Domaines référents", format_number(backlinks.get("live_refdomains", 0))],
            ["Backlinks DoFollow", format_number(backlinks.get("dofollow", 0))],
            ["Backlinks NoFollow", format_number(backlinks.get("nofollow", 0))],
        ]
        add_styled_table(doc, bl_headers, bl_rows, [6, 6])
    else:
        doc.add_paragraph("Données non disponibles (clé API Ahrefs requise).")

    doc.add_paragraph("")

    # --- 6. GSC ---
    doc.add_heading("6. Google Search Console", level=1)

    gsc_data = report.get("gsc", {})
    if gsc_data.get("available"):
        gsc_current = gsc_data.get("current", {})
        gsc_previous = gsc_data.get("previous", {})

        gsc_headers = ["Métrique", "Période actuelle", "Période précédente", "Évolution"]
        gsc_rows = []
        for metric_label, metric_key in [
            ("Clics", "clicks"), ("Impressions", "impressions"),
            ("CTR moyen", "ctr"), ("Position moyenne", "position"),
        ]:
            cur = gsc_current.get(metric_key, 0)
            prev = gsc_previous.get(metric_key, 0)
            if metric_key == "ctr":
                cur_str = f"{cur * 100:.1f}%"
                prev_str = f"{prev * 100:.1f}%"
            elif metric_key == "position":
                cur_str = f"{cur:.1f}"
                prev_str = f"{prev:.1f}"
            else:
                cur_str = format_number(cur)
                prev_str = format_number(prev)
            d, _ = delta_str(cur, prev)
            gsc_rows.append([metric_label, cur_str, prev_str, d])

        add_styled_table(doc, gsc_headers, gsc_rows, [4, 3.5, 3.5, 3])

        # Top requêtes GSC
        gsc_queries = gsc_data.get("top_queries", [])
        if gsc_queries:
            doc.add_paragraph("")
            doc.add_heading("Top requêtes GSC", level=2)
            q_headers = ["Requête", "Clics", "Impressions", "CTR", "Position"]
            q_rows = []
            for q in gsc_queries[:20]:
                keys = q.get("keys", [""])
                q_rows.append([
                    keys[0] if keys else "",
                    format_number(q.get("clicks", 0)),
                    format_number(q.get("impressions", 0)),
                    f"{q.get('ctr', 0) * 100:.1f}%",
                    f"{q.get('position', 0):.1f}",
                ])
            add_styled_table(doc, q_headers, q_rows, [5, 2, 2.5, 2, 2.5])
    else:
        doc.add_paragraph("GSC non connecté. Connectez les credentials GSC pour enrichir le rapport.")

    doc.add_paragraph("")

    # --- 7. Recommandations ---
    doc.add_heading("7. Synthèse & Recommandations", level=1)

    recommendations = report.get("recommendations", "")
    if recommendations:
        for line in recommendations.split("\n"):
            line = line.strip()
            if not line:
                doc.add_paragraph("")
            elif line.startswith("## "):
                doc.add_heading(line.replace("## ", ""), level=2)
            elif line.startswith("### "):
                doc.add_heading(line.replace("### ", ""), level=3)
            elif line.startswith("- ") or line.startswith("* "):
                doc.add_paragraph(line[2:], style="List Bullet")
            elif line.startswith("**") and line.endswith("**"):
                p = doc.add_paragraph()
                run = p.add_run(line.replace("**", ""))
                run.bold = True
            else:
                doc.add_paragraph(line)
    else:
        doc.add_paragraph("Recommandations non générées.")

    # --- Footer ---
    doc.add_page_break()
    footer = doc.add_paragraph("Rapport généré automatiquement par Ma Toolbox SEO")
    footer.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(150, 150, 150)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


# =============================================================================
# CONSTANTES
# =============================================================================
LOCATIONS = {
    "France": {"code": 2250, "lang": "fr", "ahrefs": "FR"},
    "Belgique": {"code": 2056, "lang": "fr", "ahrefs": "BE"},
    "Suisse": {"code": 2756, "lang": "fr", "ahrefs": "CH"},
    "Canada (FR)": {"code": 2124, "lang": "fr", "ahrefs": "CA"},
    "États-Unis": {"code": 2840, "lang": "en", "ahrefs": "US"},
    "Royaume-Uni": {"code": 2826, "lang": "en", "ahrefs": "GB"},
    "Allemagne": {"code": 2276, "lang": "de", "ahrefs": "DE"},
    "Espagne": {"code": 2724, "lang": "es", "ahrefs": "ES"},
    "Italie": {"code": 2380, "lang": "it", "ahrefs": "IT"},
}


# =============================================================================
# INTERFACE
# =============================================================================
st.title("📊 Rapport Client Automatisé")
st.markdown("*Génère un rapport SEO mensuel complet en un clic. GSC + Ahrefs + Claude.*")

# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Configuration API")

    anthropic_key = st.text_input(
        "Clé API Claude (Anthropic)",
        value=st.secrets.get("ANTHROPIC_API_KEY", ""),
        type="password"
    )
    ahrefs_token = st.text_input(
        "Token API Ahrefs",
        value=st.secrets.get("AHREFS_API_TOKEN", ""),
        type="password"
    )
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
    st.header("📊 Paramètres du rapport")

    # Client selection
    clients = load_clients()
    client_names = list(clients.keys())

    if client_names:
        selected_client = st.selectbox("Client", ["— Saisie manuelle —"] + client_names)
    else:
        selected_client = "— Saisie manuelle —"

    if selected_client == "— Saisie manuelle —":
        target_domain = st.text_input("Domaine", placeholder="monsite.fr")
        client_display_name = st.text_input("Nom du client", placeholder="Mon Client")
    else:
        client_data = clients[selected_client]
        target_domain = client_data.get("domain", "")
        client_display_name = client_data.get("name", selected_client)
        st.markdown(f"**Domaine** : `{target_domain}`")

    selected_location = st.selectbox("Pays cible", list(LOCATIONS.keys()), index=0)

    report_period = st.selectbox("Période du rapport", [
        "Dernier mois",
        "3 derniers mois",
        "6 derniers mois",
    ], index=0)

    gsc_property = st.text_input(
        "Propriété GSC (optionnel)",
        placeholder="sc-domain:monsite.fr ou https://www.monsite.fr/",
        help="Laisse vide si GSC n'est pas connecté"
    )

    st.divider()
    st.markdown("""
    ### Sources de données

    | Source | Données |
    |--------|---------|
    | **Ahrefs** | DR, trafic, keywords, backlinks, top pages |
    | **GSC** | Clics, impressions, CTR, positions, requêtes |
    | **Claude** | Synthèse et recommandations |

    Le rapport est généré en `.docx` prêt à envoyer au client.
    """)


# ─── Validations ───
if not anthropic_key:
    st.warning("Configure ta clé API Claude dans la sidebar.")
    st.stop()

if not ahrefs_token:
    st.warning("Configure ton token Ahrefs dans la sidebar.")
    st.stop()

if not target_domain:
    st.info("Sélectionne un client ou entre un domaine dans la sidebar.")
    st.stop()

domain = clean_domain(target_domain)
st.markdown(f"**Client** : {client_display_name} | **Domaine** : `{domain}` | **Pays** : {selected_location}")


if st.button("🚀 Générer le rapport", type="primary", use_container_width=True):
    loc = LOCATIONS[selected_location]
    ahrefs = AhrefsAPI(ahrefs_token)

    # Dates
    today = datetime.now()
    end_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")

    period_days = {"Dernier mois": 30, "3 derniers mois": 90, "6 derniers mois": 180}[report_period]
    start_date = (today - timedelta(days=period_days + 3)).strftime("%Y-%m-%d")

    prev_end = (today - timedelta(days=period_days + 3)).strftime("%Y-%m-%d")
    prev_start = (today - timedelta(days=period_days * 2 + 3)).strftime("%Y-%m-%d")

    history_start = (today - timedelta(days=365)).strftime("%Y-%m-%d")

    period_label = f"{(today - timedelta(days=period_days + 3)).strftime('%d/%m/%Y')} — {(today - timedelta(days=3)).strftime('%d/%m/%Y')}"

    report_data = {
        "client_name": client_display_name,
        "domain": domain,
        "period": period_label,
    }

    with st.status("Génération du rapport en cours...", expanded=True) as status:

        # ── Phase 1 : Ahrefs ──
        st.write("📊 **Phase 1** — Collecte Ahrefs...")

        # Domain Rating
        dr_data = ahrefs.get_domain_rating(domain, end_date)
        domain_rating = None
        if dr_data:
            domain_rating = dr_data.get("domain_rating") or dr_data.get("ahrefs_rank")
            if isinstance(dr_data, dict):
                for v in dr_data.values():
                    if isinstance(v, dict) and "domain_rating" in v:
                        domain_rating = v["domain_rating"]
                        break
                    elif isinstance(v, (int, float)) and domain_rating is None:
                        domain_rating = v
        st.write(f"  ✅ Domain Rating : {domain_rating or 'N/A'}")

        # Metrics current
        metrics_current = ahrefs.get_metrics(domain, end_date)
        current_traffic = 0
        current_keywords = 0
        if metrics_current:
            if isinstance(metrics_current, dict):
                for v in metrics_current.values():
                    if isinstance(v, dict):
                        current_traffic = v.get("org_traffic", 0) or 0
                        current_keywords = v.get("org_keywords", 0) or 0
                        break
                    elif isinstance(v, list) and v:
                        current_traffic = v[0].get("org_traffic", 0) or 0
                        current_keywords = v[0].get("org_keywords", 0) or 0
                        break

        # Metrics previous
        metrics_prev = ahrefs.get_metrics(domain, prev_end)
        prev_traffic = 0
        prev_keywords = 0
        if metrics_prev:
            if isinstance(metrics_prev, dict):
                for v in metrics_prev.values():
                    if isinstance(v, dict):
                        prev_traffic = v.get("org_traffic", 0) or 0
                        prev_keywords = v.get("org_keywords", 0) or 0
                        break
                    elif isinstance(v, list) and v:
                        prev_traffic = v[0].get("org_traffic", 0) or 0
                        prev_keywords = v[0].get("org_keywords", 0) or 0
                        break

        traffic_delta, traffic_dir = delta_str(current_traffic, prev_traffic)
        keywords_delta, kw_dir = delta_str(current_keywords, prev_keywords)

        st.write(f"  ✅ Trafic organique : {format_number(current_traffic)} ({traffic_delta})")
        st.write(f"  ✅ Mots-clés : {format_number(current_keywords)} ({keywords_delta})")

        # Backlinks stats
        bl_stats = ahrefs.get_backlinks_stats(domain, end_date)
        backlinks_data = {}
        if bl_stats:
            if isinstance(bl_stats, dict):
                for v in bl_stats.values():
                    if isinstance(v, dict) and "live" in v:
                        backlinks_data = {
                            "domain_rating": domain_rating,
                            "live_backlinks": v.get("live", 0),
                            "live_refdomains": v.get("live_refdomains", 0),
                            "dofollow": v.get("dofollow", 0),
                            "nofollow": v.get("nofollow", 0),
                        }
                        break
        if not backlinks_data and bl_stats:
            backlinks_data = {
                "domain_rating": domain_rating,
                "live_backlinks": bl_stats.get("live", 0) if isinstance(bl_stats, dict) else 0,
                "live_refdomains": bl_stats.get("live_refdomains", 0) if isinstance(bl_stats, dict) else 0,
                "dofollow": bl_stats.get("dofollow", 0) if isinstance(bl_stats, dict) else 0,
                "nofollow": bl_stats.get("nofollow", 0) if isinstance(bl_stats, dict) else 0,
            }
        st.write(f"  ✅ Backlinks : {format_number(backlinks_data.get('live_backlinks', 0))} | Refdomains : {format_number(backlinks_data.get('live_refdomains', 0))}")

        # Metrics history (12 mois)
        history_raw = ahrefs.get_metrics_history(domain, history_start, end_date)
        traffic_history = []
        if history_raw:
            rows = AhrefsAPI._rows(history_raw)
            for row in rows:
                traffic_history.append({
                    "date": row.get("date", "")[:7],
                    "traffic": row.get("org_traffic", 0) or 0,
                    "keywords": row.get("org_keywords", 0) or 0,
                    "cost": row.get("org_cost", 0) or 0,
                })
        st.write(f"  ✅ Historique : {len(traffic_history)} mois")

        # Top pages
        top_pages_raw = ahrefs.get_top_pages(domain, end_date, country=loc["ahrefs"], limit=15)
        top_pages = []
        if top_pages_raw:
            rows = AhrefsAPI._rows(top_pages_raw)
            for row in rows:
                top_pages.append({
                    "url": row.get("url", ""),
                    "traffic": row.get("sum_traffic", 0) or 0,
                    "top_keyword": row.get("top_keyword", ""),
                    "position": row.get("top_keyword_best_position", 0) or 0,
                    "keywords_count": row.get("keywords", 0) or 0,
                })

        st.write(f"  ✅ Top pages : {len(top_pages)}")

        # Organic keywords
        kw_raw = ahrefs.get_organic_keywords(domain, end_date, country=loc["ahrefs"], limit=30)
        keywords_list = []
        if kw_raw:
            rows = AhrefsAPI._rows(kw_raw)
            for row in rows:
                keywords_list.append({
                    "keyword": row.get("keyword", ""),
                    "volume": row.get("volume", 0) or 0,
                    "position": row.get("best_position", 0) or 0,
                    "position_diff": row.get("best_position_diff", 0) or 0,
                    "traffic": row.get("sum_traffic", 0) or 0,
                    "difficulty": row.get("keyword_difficulty", 0) or 0,
                    "url": row.get("best_position_url", ""),
                })

        st.write(f"  ✅ Mots-clés récupérés : {len(keywords_list)}")

        # ── Phase 2 : GSC ──
        st.write("📈 **Phase 2** — Google Search Console...")

        gsc = GSCAPI()
        gsc_report = {"available": False}

        if gsc.is_configured and gsc_property:
            # Current period
            gsc_current_raw = gsc.performance_totals(gsc_property, start_date, end_date)
            gsc_prev_raw = gsc.performance_totals(gsc_property, prev_start, prev_end)
            gsc_queries_raw = gsc.search_analytics(gsc_property, ["query"], start_date, end_date, row_limit=20)

            if gsc_current_raw:
                gsc_report = {
                    "available": True,
                    "current": gsc_current_raw,
                    "previous": gsc_prev_raw or {},
                    "top_queries": gsc_queries_raw or [],
                }
                st.write(f"  ✅ GSC connecté — Clics : {format_number(gsc_current_raw.get('clicks', 0))}")
            else:
                st.write("  ⚠️ Pas de données GSC pour cette période")
        else:
            st.write("  ℹ️ GSC non configuré — section ignorée")

        # ── Phase 3 : Recommandations Claude ──
        st.write("🧠 **Phase 3** — Synthèse et recommandations (Claude)...")

        top_pages_summary = "\n".join([
            f"- {p['url'][:60]} — trafic: {p['traffic']}, KW: {p['top_keyword']} (pos {p['position']})"
            for p in top_pages[:10]
        ]) if top_pages else "Non disponible"

        kw_movements = []
        for kw in keywords_list[:15]:
            diff = kw.get("position_diff", 0) or 0
            arrow = f"↑{diff}" if diff > 0 else f"↓{abs(diff)}" if diff < 0 else "="
            kw_movements.append(f"- {kw['keyword']} (vol:{kw['volume']}, pos:{kw['position']}, mvt:{arrow})")
        keywords_summary = "\n".join(kw_movements) if kw_movements else "Non disponible"

        recommendations_input = {
            "client_name": client_display_name,
            "domain": domain,
            "current_traffic": format_number(current_traffic),
            "traffic_delta": traffic_delta,
            "current_keywords": format_number(current_keywords),
            "keywords_delta": keywords_delta,
            "domain_rating": domain_rating,
            "refdomains": format_number(backlinks_data.get("live_refdomains", 0)),
            "top_pages_summary": top_pages_summary,
            "keywords_summary": keywords_summary,
            "gsc_clicks": format_number(gsc_report.get("current", {}).get("clicks", 0)) if gsc_report["available"] else "N/A",
            "gsc_impressions": format_number(gsc_report.get("current", {}).get("impressions", 0)) if gsc_report["available"] else "N/A",
            "gsc_ctr": f"{gsc_report.get('current', {}).get('ctr', 0) * 100:.1f}%" if gsc_report["available"] else "N/A",
            "gsc_position": f"{gsc_report.get('current', {}).get('position', 0):.1f}" if gsc_report["available"] else "N/A",
        }

        try:
            recommendations = generate_recommendations(anthropic_key, recommendations_input)
            st.write("  ✅ Recommandations générées")
        except Exception as e:
            recommendations = f"Erreur lors de la génération : {e}"
            st.write(f"  ⚠️ Erreur Claude : {e}")

        # ── Assemblage du rapport ──
        st.write("📝 **Phase 4** — Assemblage du rapport .docx...")

        kpis = {
            "Trafic organique": {
                "current": format_number(current_traffic),
                "previous": format_number(prev_traffic),
                "delta": traffic_delta,
            },
            "Mots-clés organiques": {
                "current": format_number(current_keywords),
                "previous": format_number(prev_keywords),
                "delta": keywords_delta,
            },
            "Domain Rating": {
                "current": domain_rating or "N/A",
                "previous": "—",
                "delta": "—",
            },
            "Domaines référents": {
                "current": format_number(backlinks_data.get("live_refdomains", 0)),
                "previous": "—",
                "delta": "—",
            },
        }

        if gsc_report["available"]:
            gsc_cur = gsc_report.get("current", {})
            gsc_prev = gsc_report.get("previous", {})
            clicks_d, _ = delta_str(gsc_cur.get("clicks", 0), gsc_prev.get("clicks", 0))
            impr_d, _ = delta_str(gsc_cur.get("impressions", 0), gsc_prev.get("impressions", 0))
            kpis["Clics GSC"] = {
                "current": format_number(gsc_cur.get("clicks", 0)),
                "previous": format_number(gsc_prev.get("clicks", 0)),
                "delta": clicks_d,
            }
            kpis["Impressions GSC"] = {
                "current": format_number(gsc_cur.get("impressions", 0)),
                "previous": format_number(gsc_prev.get("impressions", 0)),
                "delta": impr_d,
            }

        full_report = {
            "client_name": client_display_name,
            "domain": domain,
            "period": period_label,
            "kpis": kpis,
            "traffic_history": traffic_history,
            "top_pages": top_pages,
            "keywords": keywords_list,
            "backlinks": backlinks_data,
            "gsc": gsc_report,
            "recommendations": recommendations,
        }

        docx_buffer = create_report_docx(full_report)

        status.update(label="Rapport généré !", state="complete", expanded=False)

    # =====================================================================
    # PREVIEW DU RAPPORT
    # =====================================================================
    st.divider()

    st.markdown('<div class="section-title">📋 Aperçu du rapport</div>', unsafe_allow_html=True)

    # KPIs
    kpi_cols = st.columns(len(kpis))
    for i, (label, data) in enumerate(kpis.items()):
        with kpi_cols[i]:
            delta_val = data["delta"]
            st.metric(label, data["current"], delta=delta_val if delta_val not in ("N/A", "—") else None)

    st.divider()

    # Graphique d'évolution trafic
    if traffic_history:
        st.subheader("📈 Évolution du trafic organique (12 mois)")
        df_hist = pd.DataFrame(traffic_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["date"], y=df_hist["traffic"],
            mode="lines+markers", name="Trafic",
            line=dict(color="#667eea", width=3),
            fill="tozeroy", fillcolor="rgba(102, 126, 234, 0.1)"
        ))
        fig.update_layout(
            height=350,
            margin=dict(t=20, b=40, l=60, r=20),
            xaxis_title="Mois", yaxis_title="Trafic organique estimé",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top pages
    if top_pages:
        st.subheader("🏆 Top pages")
        df_tp = pd.DataFrame(top_pages[:10])[["url", "traffic", "top_keyword", "position"]]
        df_tp.columns = ["URL", "Trafic", "Top mot-clé", "Position"]
        st.dataframe(df_tp, use_container_width=True, hide_index=True)

    # Keywords
    if keywords_list:
        st.subheader("🔑 Mots-clés principaux")
        df_kw = pd.DataFrame(keywords_list[:15])
        df_kw["mouvement"] = df_kw["position_diff"].apply(
            lambda d: f"↑ +{d}" if d and d > 0 else f"↓ {d}" if d and d < 0 else "="
        )
        df_display = df_kw[["keyword", "volume", "position", "mouvement", "traffic"]].copy()
        df_display.columns = ["Mot-clé", "Volume", "Position", "Mouvement", "Trafic"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Recommandations
    if recommendations:
        st.subheader("💡 Recommandations")
        st.markdown(recommendations)

    st.divider()

    # ── Download ──
    filename = f"rapport_seo_{domain.replace('.', '_')}_{datetime.now().strftime('%Y%m')}.docx"
    st.download_button(
        "📥 Télécharger le rapport .docx",
        data=docx_buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
        use_container_width=True,
    )

    st.caption(f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")

st.caption("📊 Rapport Client Automatisé | Ma Toolbox SEO")
