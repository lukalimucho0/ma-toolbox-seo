import streamlit as st
from utils.auth import check_password
import anthropic
import requests
import xml.etree.ElementTree as ET
import os
import json
import re

st.set_page_config(
    page_title="Casa Barbara · SEO",
    page_icon="🏡",
    layout="wide",
)

check_password()

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #c9a84c, #f0d080);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c9a84c, #e8c96d) !important;
    color: #0d0d1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(201,168,76,0.4) !important;
}
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #c9a84c;
    margin-bottom: 0.4rem;
}
.stat-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-green { background: rgba(72,200,120,0.15); color: #48c878; border: 1px solid rgba(72,200,120,0.3); }
.badge-orange { background: rgba(255,165,0,0.15); color: #ffa500; border: 1px solid rgba(255,165,0,0.3); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STYLE GUIDE & BRAND KNOWLEDGE
# ─────────────────────────────────────────────
STYLE_EXAMPLES = """
--- ARTICLE 1 ---
Titre : Pourquoi Nice est la ville idéale pour vivre après 60 ans

Oubliez les clichés sur la retraite.

Vivre à Nice après 60 ans, ce n'est pas ralentir. C'est accélérer. C'est choisir le soleil plutôt que la grisaille. Le marché du Cours Saleya plutôt que le centre commercial. La Promenade des Anglais plutôt que le canapé.

Nice n'est pas une ville où l'on s'installe pour attendre. C'est une ville où l'on s'installe pour vivre. Pleinement. Joyeusement. Librement.

Chez Casa Barbara, on connaît cette énergie. On la vit chaque jour avec notre tribu de seniors amoureux de la vie. Alors on a décidé de vous expliquer, concrètement, pourquoi Nice est sans doute le meilleur endroit en France pour vivre après 60 ans.

1. Un climat qui change tout au quotidien
300 jours de soleil par an. Ce n'est pas un slogan, c'est une réalité. Et quand on a 60 ans ou plus, le climat n'est pas un détail : c'est un pilier de bien-être.
[...article continue...]
Rejoignez la tribu. #ForeverYoung

--- ARTICLE 2 ---
Titre : Les bienfaits du climat méditerranéen sur la santé des seniors

Et si le meilleur médicament du monde était gratuit, disponible 300 jours par an, et qu'il suffisait de sortir pour en profiter ?

Le climat méditerranéen n'est pas qu'une carte postale. C'est un véritable atout santé, documenté par des décennies de recherche scientifique.

Chez Casa Barbara, on le vit au quotidien. Nos membres le disent souvent : depuis qu'ils vivent à Nice, ils dorment mieux, bougent plus, et sourient davantage.

1. Le soleil : votre allié vitamine D au quotidien
Après 60 ans, la carence en vitamine D touche près de 80 % des Français en hiver, selon l'Inserm.
[...article continue...]
➡ Lire aussi : Pourquoi Nice est la ville idéale pour vivre après 60 ans
[...article continue...]
Rejoignez la tribu. #ForeverYoung
🏠 Découvrez Casa Barbara — Réservez votre visite gratuite
"""

BRAND_FACTS = """
FAITS CASA BARBARA (à intégrer naturellement si pertinent) :
- Résidence senior haut de gamme à Nice, quartier Saint-Roch
- Créée par la famille Trigano (après Club Med et Mama Shelter)
- Restaurant Le Barbara, imaginé avec le chef trois étoiles Pierre Gagnaire
- Concept : ni hôtel, ni maison de retraite, ni résidence classique — une "maison club"
- Activités : pilates, yoga, café philo, cours de danse, salle Technogym
- Rooftop avec vue, arrêt tramway devant la porte (réseau Lignes d'Azur)
- Séjours flexibles : une semaine, un mois ou plus longtemps
- Hashtag : #ForeverYoung | Phrase de clôture : "Rejoignez la tribu."
- CTA final : "🏠 Découvrez Casa Barbara — Réservez votre visite gratuite"
"""

STYLE_RULES = """
RÈGLES DE STYLE ABSOLUES :
1. Ton : direct, punchy, anti-clichés sur la vieillesse. Jamais condescendant.
2. Voix : "Chez Casa Barbara, on..." — 1ère personne du pluriel pour la marque.
3. Phrases courtes. Très courtes parfois. C'est voulu.
4. Accroche en rupture : commence par une idée provocatrice ou contre-intuitive.
5. Structure : intro forte → sections numérotées (H2) → sous-sections (H3 si besoin) → conclusion CTA.
6. Données : citer INSERM, OMS, études si pertinent pour crédibiliser.
7. Liens internes : format exact → <p>➡ <a href="URL">Lire aussi : Titre</a></p>
8. Clôture TOUJOURS : <p>Rejoignez la tribu. #ForeverYoung</p> puis CTA Casa Barbara.
9. Vocabulaire : "tribu", "liberté", "joie", "vitalité", "bien vivre", "élan", "seniors".
10. INTERDITS ABSOLUS :
    - "personnes âgées" (→ toujours "seniors")
    - Langage médical froid et tournures passéistes
    - Les tirets "-" comme ponctuation dans les phrases (utilise la virgule, le point ou les deux-points à la place)
    - Les balises <hr> et tout séparateur visuel entre paragraphes
    - Les listes à puces pour le contenu principal (réserve <ul>/<li> uniquement pour des énumérations très courtes et factuelles)
"""

SYSTEM_PROMPT = f"""Tu es le rédacteur officiel de Casa Barbara, résidence senior haut de gamme à Nice.
Ta mission : créer des articles SEO qui capturent parfaitement le style Casa Barbara — chaleureux, direct, élégant, résolument anti-clichés sur le vieillissement.

{STYLE_RULES}

{BRAND_FACTS}

EXEMPLES DE STYLE À IMITER :
{STYLE_EXAMPLES}
"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _extract_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


SITEMAP_URLS = [
    "https://www.casabarbara.com/post-sitemap.xml",
    "https://www.casabarbara.com/page-sitemap.xml",
]


@st.cache_data(ttl=86400, show_spinner=False)
def get_sitemap_urls() -> list:
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    headers = {"User-Agent": "Mozilla/5.0"}
    all_urls = []
    for sitemap_url in SITEMAP_URLS:
        try:
            r = requests.get(sitemap_url, timeout=10, headers=headers)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            all_urls.extend(
                loc.text.strip() for loc in root.findall(".//sm:loc", ns) if loc.text
            )
        except Exception as e:
            st.warning(f"Sitemap inaccessible ({sitemap_url}) : {e}")
    return all_urls


def analyze_keyword(keyword: str, urls: list, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    urls_text = "\n".join(urls[:80]) if urls else "Aucune URL"

    prompt = f"""Analyse ce mot-clé SEO pour Casa Barbara (résidence senior haut de gamme à Nice).

Mot-clé cible : "{keyword}"

URLs du sitemap casabarbara.com :
{urls_text}

Retourne UNIQUEMENT un JSON valide :
{{
  "intent": "informationnel|transactionnel|navigationnel|commercial",
  "audience": "description précise de qui recherche ce terme",
  "semantic_field": ["thème 1", "thème 2", "..."],
  "structure": [
    {{"h2": "Titre section", "h3s": ["sous-section a", "sous-section b"]}},
    {{"h2": "Autre section", "h3s": []}}
  ],
  "internal_links": [
    {{"url": "url-exacte-du-sitemap", "anchor": "texte d'ancre naturel"}}
  ]
}}

Consignes :
- 5-8 sections H2 logiques, exhaustives, SEO-friendly
- Champ sémantique : couvrir toutes les entités et sous-thèmes liés
- 3-5 liens internes issus du sitemap, sélectionnés thématiquement
- Audience : seniors 60+ ou leur famille cherchant une résidence à Nice"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return {
        "intent": "informationnel",
        "audience": "Seniors 60+ ou leur famille",
        "semantic_field": [keyword],
        "structure": [{"h2": f"Tout savoir sur {keyword}", "h3s": []}],
        "internal_links": [],
    }


def generate_article(keyword: str, analysis: dict, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)

    structure_text = "\n".join(
        f"H2 : {s['h2']}" + (
            "\n" + "\n".join(f"  H3 : {h}" for h in s.get("h3s", []))
            if s.get("h3s") else ""
        )
        for s in analysis.get("structure", [])
    )

    links_text = "\n".join(
        f'- <a href="{l["url"]}">{l["anchor"]}</a>'
        for l in analysis.get("internal_links", [])
    ) or "Aucun lien disponible"

    prompt = f"""Génère un article SEO complet pour Casa Barbara sur : "{keyword}"

CONTEXTE :
- Intention : {analysis.get("intent", "informationnel")}
- Audience : {analysis.get("audience", "Seniors 60+ et leur famille")}
- Champ sémantique : {", ".join(analysis.get("semantic_field", []))}

STRUCTURE À SUIVRE :
{structure_text}

LIENS INTERNES (format : <p>➡ <a href="url">Lire aussi : ancre</a></p>) :
{links_text}

FORMAT DE RÉPONSE — respecte exactement ces balises :
<title>Title SEO · 60-65 caractères max</title>
<meta>Meta description engageante · 150-160 caractères max</meta>
<slug>slug-en-minuscules-sans-accents-avec-tirets</slug>
<html>
[article HTML complet — balises h1 h2 h3 p strong ul li — sans html/head/body/style]
</html>

IMPÉRATIFS HTML :
- <h1> accrocheur (peut différer du title SEO)
- Sections numérotées en <h2> selon la structure
- Paragraphes courts, rythmés, style Casa Barbara
- Liens internes intégrés naturellement
- Terminer TOUJOURS : <p>Rejoignez la tribu. #ForeverYoung</p> + CTA Casa Barbara
- Longueur : couvre tout le champ sémantique, sans limite fixe"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    return {
        "title": _extract_tag(text, "title"),
        "meta_description": _extract_tag(text, "meta"),
        "slug": _extract_tag(text, "slug"),
        "html_body": _extract_tag(text, "html"),
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Clé API Anthropic",
        type="password",
        value=st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")),
        placeholder="sk-ant-...",
    )
    st.divider()
    st.markdown("### 🗺️ Sitemap")
    if st.button("Actualiser le cache", use_container_width=True):
        get_sitemap_urls.clear()
        st.success("Cache vidé !")
    st.caption("Sitemap mis en cache 24h")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("Casa Barbara · Générateur SEO")
st.markdown(
    "<span style='color:#a09070;'>Style maison intégré · Maillage interne automatique · Claude Sonnet</span>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
col_in, col_btn = st.columns([4, 1])
with col_in:
    keyword = st.text_input(
        "Mot-clé",
        placeholder="ex: résidence senior Nice, activités seniors bord de mer...",
        label_visibility="collapsed",
    )
with col_btn:
    generate = st.button(
        "🚀 Générer",
        type="primary",
        use_container_width=True,
        disabled=not keyword or not api_key,
    )

if not api_key:
    st.info("👈 Renseigne ta clé API Anthropic dans la sidebar.")
    st.stop()

if not generate:
    st.stop()

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
with st.status("Génération en cours...", expanded=True) as status:
    st.write("📡 Récupération du sitemap casabarbara.com...")
    urls = get_sitemap_urls()
    st.write(f"✅ {len(urls)} URLs chargées")

    st.write("🔍 Analyse intention & champ sémantique...")
    try:
        analysis = analyze_keyword(keyword, urls, api_key)
        st.write("✅ Analyse terminée")
    except Exception as e:
        st.error(f"Erreur analyse : {e}")
        st.stop()

    st.write("✍️ Rédaction en style Casa Barbara...")
    try:
        result = generate_article(keyword, analysis, api_key)
        st.write("✅ Article prêt !")
    except Exception as e:
        st.error(f"Erreur génération : {e}")
        st.stop()

status.update(label="Article généré !", state="complete", expanded=False)

# ─────────────────────────────────────────────
# ANALYSE
# ─────────────────────────────────────────────
with st.expander("🔎 Analyse SEO détectée", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Intention :** `{analysis.get('intent', 'N/A')}`")
        st.markdown(f"**Audience :** {analysis.get('audience', 'N/A')}")
        if analysis.get("internal_links"):
            st.markdown("**Liens internes sélectionnés :**")
            for link in analysis["internal_links"]:
                st.markdown(f"• [{link.get('anchor', '')}]({link.get('url', '')})")
    with c2:
        st.markdown("**Champ sémantique :**")
        for item in analysis.get("semantic_field", []):
            st.markdown(f"• {item}")

st.divider()

# ─────────────────────────────────────────────
# MÉTADONNÉES
# ─────────────────────────────────────────────
st.markdown("### Métadonnées SEO")
c1, c2, c3 = st.columns([5, 6, 3])

title = result.get("title", "")
meta = result.get("meta_description", "")
slug = result.get("slug", "")

with c1:
    st.markdown('<div class="section-label">Title</div>', unsafe_allow_html=True)
    st.code(title, language=None)
    t_len = len(title)
    badge = "badge-green" if t_len <= 65 else "badge-orange"
    st.markdown(f'<span class="stat-badge {badge}">{t_len} / 65 car.</span>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-label">Meta Description</div>', unsafe_allow_html=True)
    st.code(meta, language=None)
    m_len = len(meta)
    badge = "badge-green" if m_len <= 160 else "badge-orange"
    st.markdown(f'<span class="stat-badge {badge}">{m_len} / 160 car.</span>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="section-label">Slug</div>', unsafe_allow_html=True)
    st.code(slug, language=None)

st.divider()

# ─────────────────────────────────────────────
# ARTICLE
# ─────────────────────────────────────────────
st.markdown("### Article")
html_body = result.get("html_body", "")

tab_html, tab_preview = st.tabs(["📋 HTML brut", "👁️ Aperçu rendu"])
with tab_html:
    st.code(html_body, language="html")
with tab_preview:
    st.markdown(
        "<div style='max-width:860px;margin:auto;line-height:1.8;'>" + html_body + "</div>",
        unsafe_allow_html=True,
    )
