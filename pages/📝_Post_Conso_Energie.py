"""
Post Conso Energie — outil tout-en-un.

Depuis un seul mot-clé, enchaîne automatiquement : recherche SERP + concurrents
(DataForSEO), sélection, structure Hn, title/meta, rédaction, puis mise en forme
« gold » (réponse en bref, infographie, FAQ + schema, sources). L'utilisateur n'a
plus qu'à relire et cliquer sur « Publier sur WP » (brouillon par défaut).

Réutilise le moteur de la page Rédaction via utils.seo_engine (sans duplication).
"""

import streamlit as st
import streamlit.components.v1 as components
from utils.auth import check_password
from utils.seo_engine import (
    SEOBriefGenerator, PromptTemplates, HeadingParser, run_writing_engine,
    DataForSEOConfig, nodes_to_markdown,
)
from utils.gold_enrich import enrich_to_gold, build_gold_html
from utils.wordpress import push_article, slugify, CATEGORIES, count_em_dash
from utils.maillage_conso import maillage_block

st.set_page_config(page_title="Post Conso Energie", page_icon="📝", layout="wide")
check_password()

st.title("📝 Post Conso Energie")
st.caption("Un mot-clé → recherche, concurrence, structure, rédaction et mise en forme gold. Tu relis, tu publies.")

CLIENT_BRIEF = (
    "conso-energie.fr est un guide national, neutre et factuel, de l'energie du logement. "
    "Ton informatif, clair et pedagogique, vouvoiement, francais soigne, sans superlatifs commerciaux. "
    "JAMAIS de tiret cadratin. Aucune allegation d'identite officielle. Les artisans sont qualifies et verifies (RGE). "
    "Objectif : informer puis orienter naturellement vers une mise en relation."
)


def _secret(key, default=""):
    v = st.secrets.get(key, "")
    if v:
        return v
    for val in st.secrets.values():
        try:
            if isinstance(val, dict) and val.get(key):
                return val.get(key)
        except Exception:
            pass
    return default


def _clean_title(raw):
    if not raw:
        return ""
    line = raw.strip().strip('"').splitlines()[0].strip()
    low = line.lower()
    if low.startswith(("title", "titre", "meta")):
        line = line.split(":", 1)[-1].strip()
    return line.strip().strip('"')


# ---------- Secrets / config ----------
WP_URL, WP_TOKEN = _secret("WP_INGEST_URL"), _secret("WP_INGEST_TOKEN")
ANTHROPIC_KEY = _secret("ANTHROPIC_API_KEY")
DFS_USER, DFS_PASS = _secret("DATAFORSEO_USERNAME"), _secret("DATAFORSEO_PASSWORD")

missing = [n for n, v in [
    ("WP_INGEST_URL", WP_URL), ("WP_INGEST_TOKEN", WP_TOKEN), ("ANTHROPIC_API_KEY", ANTHROPIC_KEY),
    ("DATAFORSEO_USERNAME", DFS_USER), ("DATAFORSEO_PASSWORD", DFS_PASS)] if not v]
if missing:
    st.error("Secrets manquants : " + ", ".join(missing) + ". Ajoute-les (au-dessus de toute section [..]).")
    st.stop()

with st.sidebar:
    st.subheader("Paramètres")
    model = st.selectbox("Modèle de rédaction", [
        "claude-sonnet-4-5-20250929", "claude-opus-4-6", "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",
    ])
    num_results = st.slider("Résultats SERP analysés", 5, 15, 10)
    cat_force = st.selectbox("Catégorie (laisser auto si possible)", ["(auto)"] + CATEGORIES)
    st.markdown("**Réservoir de maillage interne**")
    st.caption("L'outil pioche dans cette liste l'ancre pertinente selon le contenu (1 lien par ancre maximum). Format : url | ancre.")
    liens = st.text_area("url | ancre (une par ligne)", height=220, value=maillage_block())


def _liens_formatted(txt):
    out = []
    for l in (txt or "").splitlines():
        if "|" in l:
            url, anchor = [p.strip() for p in l.split("|", 1)]
            if url and anchor:
                out.append(f"- [{anchor}]({url})")
    if not out:
        return ""
    return "LIENS INTERNES À PLACER (format markdown [ancre](url), de façon naturelle) :\n" + "\n".join(out)


# ---------- Saisie + lancement ----------
keyword = st.text_input("Mot-clé cible", placeholder="Ex. comment entretenir une pompe à chaleur")
go = st.button("🚀 Lancer la génération complète", type="primary", disabled=not keyword.strip())

if go:
    try:
        gen = SEOBriefGenerator()
        gen.setup_apis(DFS_USER, DFS_PASS, "claude", model, ANTHROPIC_KEY)
        country, language = "🇫🇷 France", "Français"
        language_code = DataForSEOConfig.get_market_config(country, language)["language_code"]
        ilf = (liens or "").strip()

        with st.status("Génération en cours...", expanded=True) as status:
            st.write("🔍 1/6 Recherche SERP et extraction des concurrents...")
            competitors = gen.search_and_extract_competitors(keyword, country, language, num_results)
            if not competitors:
                status.update(label="Échec", state="error")
                st.error("Aucun concurrent trouvé. Vérifie les identifiants DataForSEO (email + clé API).")
                st.stop()
            st.write(f"→ {len(competitors)} concurrents analysés")

            st.write("🤖 2/6 Sélection des meilleurs concurrents...")
            selected = gen.auto_select_competitors(competitors, max_competitors=5)
            st.write(f"→ {len(selected)} retenus")

            st.write("🏗️ 3/6 Construction de la structure Hn...")
            structure = gen.ai_analyzer.analyze_with_custom_prompt(
                PromptTemplates.get_structure_prompt(keyword, selected), max_tokens=3000)
            nodes = HeadingParser.parse_structure_text(structure)
            st.write(f"→ {len(nodes)} sections")

            st.write("🏷️ 4/6 Title et meta description...")
            title = gen.ai_analyzer.analyze_with_custom_prompt(
                PromptTemplates.get_title_prompt(keyword, selected), max_tokens=200)
            meta = gen.ai_analyzer.analyze_with_custom_prompt(
                PromptTemplates.get_meta_description_prompt(keyword, selected), max_tokens=300)

            st.write("✍️ 5/6 Rédaction de l'article (section par section)...")
            article_nodes = run_writing_engine(
                nodes=nodes, target_keyword=keyword, selected_competitors=selected,
                ai_analyzer=gen.ai_analyzer, language_code=language_code,
                internal_links_formatted=ilf, client_brief=CLIENT_BRIEF,
                status_callback=lambda s: st.write(s),
            )
            full_md = nodes_to_markdown(article_nodes)

            st.write("✨ 6/6 Mise en forme gold + métadonnées...")
            enrich = enrich_to_gold(gen.ai_analyzer, keyword, full_md)
            gold_html = build_gold_html(full_md, enrich)
            status.update(label="✅ Article prêt à publier", state="complete")

        st.session_state.pce = {
            "html": gold_html, "enrich": enrich, "title": _clean_title(title),
            "meta": _clean_title(meta), "md": full_md, "keyword": keyword,
            "cat_force": cat_force,
        }
    except Exception as e:
        st.error(f"Erreur pendant la génération : {e}")


# ---------- Résultat + publication ----------
pce = st.session_state.get("pce")
if pce:
    st.divider()
    st.subheader("Résultat")
    enrich = pce["enrich"]
    gold_html = pce["html"]

    c1, c2 = st.columns([2, 1])
    with c2:
        titre = st.text_input("Titre", value=pce["title"], key="pce_titre")
        slug = st.text_input("Slug", value=enrich.get("slug") or slugify(titre), key="pce_slug")
        cats = CATEGORIES
        if pce.get("cat_force") in cats:
            default_cat = pce["cat_force"]
        else:
            default_cat = enrich.get("category") if enrich.get("category") in cats else cats[0]
        categorie = st.selectbox("Catégorie", cats, index=cats.index(default_cat), key="pce_cat")
        meta = st.text_area("Meta description", value=enrich.get("metadesc") or pce["meta"] or "", height=90, key="pce_meta")
        statut = st.radio("Statut", ["draft", "publish"], horizontal=True, key="pce_statut",
                          format_func=lambda s: "Brouillon" if s == "draft" else "En ligne")
        overwrite = st.checkbox("Écraser si le slug existe", key="pce_ov")
        st.metric("Tirets cadratins", count_em_dash(gold_html))
        st.metric("Mots (corps)", len(pce["md"].split()))

    with c1:
        preview_css = (
            "<style>.cep{font-family:system-ui;max-width:720px}.cep .tldr{background:#f3f3ff;border-left:4px solid #000091;"
            "padding:12px 16px;border-radius:4px}.cep h2{color:#000091}.cep .infographie{border:1px solid #dcdce6;border-radius:6px;padding:12px}"
            ".cep table{border-collapse:collapse;width:100%}.cep th{background:#000091;color:#fff;padding:6px}.cep td{border-bottom:1px solid #dcdce6;padding:6px}"
            ".cep details{border:1px solid #dcdce6;border-radius:4px;margin-bottom:6px;padding:8px}.cep .sources{font-size:.85rem;color:#525275}</style>"
        )
        components.html(preview_css + "<div class='cep'>" + gold_html + "</div>", height=560, scrolling=True)

    with st.expander("Voir le markdown brut"):
        st.code(pce["md"], language="markdown")

    if st.button("🚀 Publier sur WP", type="primary", key="pce_push"):
        try:
            res = push_article(
                WP_URL, WP_TOKEN, title=titre, content_html=gold_html, category=categorie,
                excerpt=meta, slug=slug, status=statut, metadesc=meta, overwrite=overwrite,
            )
            verbe = "mis à jour" if res.get("updated") else "créé"
            st.success(f"✅ Article {verbe} ({res.get('status')}) sur WordPress, id {res.get('id')}")
            st.markdown(f"[Voir l'article →]({res.get('link')})")
        except RuntimeError as e:
            st.error(str(e))
