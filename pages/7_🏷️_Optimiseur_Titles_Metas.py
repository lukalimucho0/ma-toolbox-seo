"""
Optimiseur Titles & Metas
Génère 3 variantes optimisées de title et meta description à partir
d'une URL + mot clé cible. Scraping parallèle page + SERP DataForSEO,
génération via Claude Sonnet 4.6.
"""

import streamlit as st
from utils.auth import check_password
import requests
import re
import base64
import json
import logging
import time
import concurrent.futures
from typing import List, Dict
from urllib.parse import urlparse

import anthropic
from bs4 import BeautifulSoup
import trafilatura
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langdetect import detect as _langdetect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════

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
        .success-box {
            background-color: #F0FDF4;
            border-left: 4px solid #22C55E;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        .char-ok   { color: #22C55E; font-weight: 600; }
        .char-warn { color: #F59E0B; font-weight: 600; }
        .char-bad  { color: #EF4444; font-weight: 600; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px 8px 0 0; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CONFIGS LANGUE / MARCHÉ
# ══════════════════════════════════════════════════════════════

LANGUAGE_CONFIGS = {
    "fr": {"location_code": 2250, "language_code": "fr", "name": "Français"},
    "en": {"location_code": 2826, "language_code": "en", "name": "English"},
    "es": {"location_code": 2724, "language_code": "es", "name": "Español"},
    "de": {"location_code": 2276, "language_code": "de", "name": "Deutsch"},
    "it": {"location_code": 2380, "language_code": "it", "name": "Italiano"},
    "pt": {"location_code": 2620, "language_code": "pt", "name": "Português"},
    "nl": {"location_code": 2528, "language_code": "nl", "name": "Nederlands"},
}

FRENCH_STOP_WORDS = {
    'le','la','les','de','du','des','un','une','et','en','pour','avec',
    'sur','par','dans','est','sont','comment','meilleur','meilleure',
    'prix','acheter','pas','cher','guide','tout','savoir','faire',
    'trouver','compare','comparatif','avis','liste'
}


# ══════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════

def extract_brand_from_url(url: str) -> str:
    """Extrait le nom de marque depuis le domaine de l'URL."""
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    domain = re.sub(r'^www\.', '', domain)
    # Retire le TLD (gère aussi .co.uk, .co.fr, etc.)
    domain = re.sub(
        r'\.(com|fr|net|org|io|co|eu|be|ch|ca|uk|de|es|it|pt|nl|pl|mx|br|ar|cl|app|store|shop)(\.[a-z]{2})?$',
        '', domain, flags=re.IGNORECASE
    )
    brand = domain.replace('-', ' ').replace('_', ' ').title()
    return brand or "Marque"


def detect_language(keyword: str) -> str:
    """Détecte la langue du mot clé (défaut : français)."""
    if HAS_LANGDETECT:
        try:
            lang = _langdetect(keyword)
            return lang if lang in LANGUAGE_CONFIGS else "fr"
        except Exception:
            pass

    kw_lower = keyword.lower()
    # Caractères accentués français
    if re.search(r'[àâäéèêëîïôùûüç]', kw_lower):
        return "fr"
    # Mots courants français
    if set(kw_lower.split()) & FRENCH_STOP_WORDS:
        return "fr"
    # Mots courants anglais
    en_words = {'the','how','best','buy','top','near','cheap','free','online','guide'}
    if set(kw_lower.split()) & en_words:
        return "en"

    return "fr"  # défaut consultant FR


def char_badge(count: int, min_v: int, max_v: int) -> str:
    if count > max_v:
        css, msg = "char-bad", f"⚠️ Trop long ({count} car.)"
    elif count < min_v:
        css, msg = "char-warn", f"⚡ Court ({count} car.)"
    else:
        css, msg = "char-ok", f"✅ {count} car."
    return f'<span class="{css}">{msg}</span>'


# ══════════════════════════════════════════════════════════════
# DATAFORSEO — SERP
# ══════════════════════════════════════════════════════════════

def fetch_serp_data(keyword: str, lang_config: dict, username: str, password: str) -> List[Dict]:
    """Récupère les top 5 résultats SERP organiques via DataForSEO."""
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json"
    }
    payload = [{
        "keyword": keyword,
        "location_code": lang_config["location_code"],
        "language_code": lang_config["language_code"],
        "device": "desktop",
        "os": "windows",
        "depth": 10,
        "calculate_rectangles": False
    }]

    try:
        resp = requests.post(
            "https://api.dataforseo.com/v3/serp/google/organic/live/regular",
            headers=headers,
            json=payload,
            timeout=15
        )
        if resp.status_code == 401:
            logger.error("DataForSEO 401 — vérifier email + clé API")
            return []
        resp.raise_for_status()
        data = resp.json()

        if data.get("status_code") != 20000:
            logger.error(f"DataForSEO: {data.get('status_message')}")
            return []

        results = []
        items = (
            data.get("tasks", [{}])[0]
            .get("result", [{}])[0]
            .get("items", [])
        )
        for item in items:
            if item.get("type") == "organic":
                results.append({
                    "position": item.get("rank_absolute", 0),
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                })
            if len(results) >= 5:
                break
        return results

    except Exception as e:
        logger.error(f"DataForSEO SERP error: {e}")
        return []


# ══════════════════════════════════════════════════════════════
# SCRAPING PAGE CIBLE
# ══════════════════════════════════════════════════════════════

def scrape_page(url: str) -> Dict:
    """Scrape l'URL cible : title, meta, H1, extrait de contenu."""
    result = {
        "current_title": "",
        "current_meta": "",
        "h1": "",
        "content_summary": "",
        "success": False
    }
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        title_tag = soup.find('title')
        result["current_title"] = title_tag.get_text(strip=True) if title_tag else ""

        meta_tag = soup.find('meta', attrs={'name': re.compile(r'^description$', re.I)})
        if meta_tag:
            result["current_meta"] = meta_tag.get('content', '')

        h1_tag = soup.find('h1')
        result["h1"] = h1_tag.get_text(strip=True) if h1_tag else ""

        content = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if content:
            result["content_summary"] = content[:600]

        result["success"] = True
    except Exception as e:
        logger.warning(f"Scraping {url}: {e}")
    return result


# ══════════════════════════════════════════════════════════════
# GÉNÉRATION CLAUDE SONNET 4.6
# ══════════════════════════════════════════════════════════════

def generate_variants(
    keyword: str,
    brand: str,
    page_data: Dict,
    serp_data: List[Dict],
    lang_code: str,
    api_key: str
) -> List[Dict]:
    """Génère 3 variantes {title, meta} via Claude Sonnet 4.6."""

    client = anthropic.Anthropic(api_key=api_key)
    lang_name = LANGUAGE_CONFIGS.get(lang_code, LANGUAGE_CONFIGS["fr"])["name"]

    # Contexte page
    page_ctx = ""
    if page_data["success"]:
        parts = []
        if page_data["current_title"]:
            parts.append(f"Title actuel : {page_data['current_title']}")
        if page_data["current_meta"]:
            parts.append(f"Meta actuelle : {page_data['current_meta']}")
        if page_data["h1"]:
            parts.append(f"H1 : {page_data['h1']}")
        if page_data["content_summary"]:
            parts.append(f"Contenu (extrait) : {page_data['content_summary']}")
        page_ctx = "\n".join(parts)

    # Contexte SERP
    serp_ctx = ""
    if serp_data:
        lines = []
        for r in serp_data:
            desc = (r["description"] or "")[:120]
            lines.append(f"#{r['position']} « {r['title']} » — {desc}")
        serp_ctx = "\n".join(lines)

    prompt = f"""Tu es un expert SEO spécialisé dans l'optimisation des balises title et meta.

MOT CLÉ CIBLE : {keyword}
MARQUE : {brand}
LANGUE DE RÉDACTION : {lang_name}

CONTEXTE DE LA PAGE :
{page_ctx or "Non disponible"}

TOP SERP ACTUELS (concurrents) :
{serp_ctx or "Non disponible"}

RÈGLES TITLE (respecter strictement) :
- Le mot clé cible commence TOUJOURS le title, tel quel, sans modification
- Format OBLIGATOIRE : {keyword} : [accroche sémantique] | {brand}
- La longueur n'a pas d'importance : Google réécrit souvent les titles
- RÈGLE CRITIQUE sur l'accroche : Google supprime souvent la partie avant ":" et n'affiche
  que "[accroche sémantique] | {brand}". L'accroche DOIT donc fonctionner seule.
  Elle doit être riche en champ sémantique du mot clé cible : utilise des synonymes,
  des mots clés secondaires, des variantes sémantiques, des termes de la même famille lexicale.
  C'est une phrase ou expression SEO-dense, pas un slogan générique.
  Exemple pour "plombier paris" : "Plombier Paris : Dépannage Urgent 24h/7j | Marque"
  → si Google ne garde que "Dépannage Urgent 24h/7j | Marque", ça reste pertinent et bien positionné.
- Chaque variante doit exploiter un angle sémantique différent (intention, synonyme, besoin)

RÈGLES META DESCRIPTION :
- 150 à 160 caractères
- Langue : {lang_name}
- Engageante, décrit précisément ce qu'on trouve sur la page
- Doit créer de la curiosité ou une forte envie de cliquer
- Pas de mensonge ni de promesse non tenue

Génère EXACTEMENT 3 variantes différentes.
Réponds UNIQUEMENT avec ce JSON valide, sans texte ni markdown autour :
[
  {{"title": "...", "meta": "..."}},
  {{"title": "...", "meta": "..."}},
  {{"title": "...", "meta": "..."}}
]"""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Optimiseur Titles & Metas | Ma Toolbox SEO",
        page_icon="🏷️",
        layout="wide"
    )
    check_password()
    inject_css()

    st.markdown('<div class="main-header">🏷️ Optimiseur Titles & Metas</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">URL + mot clé → 3 variantes optimisées générées en quelques secondes</div>',
        unsafe_allow_html=True
    )

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🔍 DataForSEO API")
        dfs_user = st.text_input(
            "Username (email)",
            value=st.secrets.get("DATAFORSEO_USERNAME", ""),
            type="password"
        )
        dfs_pass = st.text_input(
            "Password (clé API)",
            value=st.secrets.get("DATAFORSEO_PASSWORD", ""),
            type="password"
        )

        st.subheader("🤖 Anthropic API")
        ant_key = st.text_input(
            "Clé API Anthropic",
            value=st.secrets.get("ANTHROPIC_API_KEY", ""),
            type="password"
        )

        st.divider()
        st.caption("**Modèle :** Claude Sonnet 4.6")
        st.caption("**Format :** `Mot clé : Accroche sémantique | Marque`")
        st.caption("**Meta :** 150–160 car. · **Title :** longueur libre")

    # ── Formulaire principal ───────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        target_url = st.text_input(
            "🌐 URL de la page",
            placeholder="https://www.mon-site.fr/ma-page/",
            help="La marque sera extraite automatiquement depuis le domaine"
        )
    with col2:
        target_keyword = st.text_input(
            "🎯 Mot clé cible",
            placeholder="ex : plombier paris, meilleur hôtel new york…",
            help="La langue est détectée automatiquement"
        )

    run = st.button("✨ Générer les optimisations", type="primary", use_container_width=True)

    if not run:
        return

    # ── Validation ────────────────────────────────────────────
    errors = []
    if not target_url:
        errors.append("URL de la page manquante.")
    if not target_keyword:
        errors.append("Mot clé cible manquant.")
    if not dfs_user or not dfs_pass:
        errors.append("Identifiants DataForSEO manquants (sidebar).")
    if not ant_key:
        errors.append("Clé Anthropic manquante (sidebar).")
    if errors:
        for e in errors:
            st.error(f"❌ {e}")
        return

    brand = extract_brand_from_url(target_url)
    lang_code = detect_language(target_keyword)
    lang_config = LANGUAGE_CONFIGS.get(lang_code, LANGUAGE_CONFIGS["fr"])

    st.markdown(
        f'<div class="insight-box">'
        f'🏷️ <strong>Marque :</strong> {brand} &nbsp;|&nbsp; '
        f'🌍 <strong>Langue :</strong> {lang_config["name"]} &nbsp;|&nbsp; '
        f'📍 <strong>Marché :</strong> location #{lang_config["location_code"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Fetch en parallèle : page + SERP ──────────────────────
    t0 = time.time()
    with st.spinner("⏳ Scraping page + SERP en parallèle…"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_page = pool.submit(scrape_page, target_url)
            f_serp = pool.submit(fetch_serp_data, target_keyword, lang_config, dfs_user, dfs_pass)
            page_data = f_page.result()
            serp_data = f_serp.result()
    fetch_time = time.time() - t0

    # Affiche état actuel de la page
    if page_data["success"]:
        with st.expander("📄 État actuel de la page", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Title actuel**")
                st.code(page_data["current_title"] or "*(non trouvé)*", language=None)
                if page_data["current_title"]:
                    n = len(page_data["current_title"])
                    st.markdown(char_badge(n, 50, 60), unsafe_allow_html=True)
            with c2:
                st.markdown("**Meta description actuelle**")
                st.code(page_data["current_meta"] or "*(non trouvée)*", language=None)
                if page_data["current_meta"]:
                    n = len(page_data["current_meta"])
                    st.markdown(char_badge(n, 150, 160), unsafe_allow_html=True)
    else:
        st.warning("⚠️ Impossible de scraper la page — la génération s'appuie uniquement sur les données SERP.")

    # Affiche SERP
    if serp_data:
        with st.expander(f"🔍 Top {len(serp_data)} SERP analysés", expanded=False):
            for r in serp_data:
                st.markdown(f"**#{r['position']}** {r['title']}")
                if r["description"]:
                    st.caption(r["description"])
                st.caption(f"↳ {r['url']}")
    else:
        st.warning("⚠️ Données SERP non disponibles — vérifier les clés DataForSEO.")

    # ── Génération Claude ──────────────────────────────────────
    with st.spinner("🤖 Génération Claude Sonnet 4.6…"):
        t_gen = time.time()
        try:
            variants = generate_variants(
                keyword=target_keyword,
                brand=brand,
                page_data=page_data,
                serp_data=serp_data,
                lang_code=lang_code,
                api_key=ant_key
            )
        except json.JSONDecodeError as e:
            st.error(f"❌ Réponse Claude non parseable : {e}")
            return
        except Exception as e:
            st.error(f"❌ Erreur Claude : {e}")
            return
        gen_time = time.time() - t_gen
        total_time = time.time() - t0

    # ── Résultats ──────────────────────────────────────────────
    st.markdown(
        f'<div class="section-title">✨ 3 variantes générées '
        f'<span style="font-size:0.85rem;color:#6B7280;font-weight:400">'
        f'— ⏱ {total_time:.1f}s total ({fetch_time:.1f}s fetch · {gen_time:.1f}s IA)'
        f'</span></div>',
        unsafe_allow_html=True
    )

    labels = ["A", "B", "C"]
    tabs = st.tabs([f"Variante {l}" for l in labels])

    for tab, label, variant in zip(tabs, labels, variants):
        with tab:
            title_val = variant.get("title", "")
            meta_val = variant.get("meta", "")
            t_len = len(title_val)
            m_len = len(meta_val)

            st.markdown("#### 📌 Title")
            st.code(title_val, language=None)
            st.caption(f"{t_len} caractères — longueur non contrainte (Google réécrit souvent)")

            st.markdown("#### 📝 Meta Description")
            st.code(meta_val, language=None)
            st.markdown(char_badge(m_len, 150, 160), unsafe_allow_html=True)

    # Tableau comparatif
    st.markdown('<div class="section-title">📊 Tableau comparatif</div>', unsafe_allow_html=True)
    rows = []
    for label, v in zip(labels, variants):
        title_val = v.get("title", "")
        meta_val = v.get("meta", "")
        rows.append({
            "Var.": label,
            "Title": title_val,
            "Meta Description": meta_val,
            "Car. meta": len(meta_val),
        })
    df = pd.DataFrame(rows)

    def style_meta_char(val):
        if val > 160:
            return "color: #EF4444; font-weight: bold"
        if val < 150:
            return "color: #F59E0B; font-weight: bold"
        return "color: #22C55E; font-weight: bold"

    styled = (
        df.style
        .applymap(style_meta_char, subset=["Car. meta"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="success-box">💡 <strong>Astuce :</strong> '
        'Cliquez sur une cellule du tableau pour copier la valeur. '
        'Les cases vertes indiquent une longueur optimale pour Google.</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
