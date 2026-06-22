"""
Page de publication WordPress.

Récupère l'article produit par la page « Rédaction Contenu » (markdown stocké dans
st.session_state.article_content), permet d'ajouter les blocs « gold » (réponse en
bref, infographie, FAQ + schema, sources), prévisualise, puis pousse l'article sur
conso-energie.fr via l'endpoint d'ingestion (base64 + token).
"""

import streamlit as st
from utils.auth import check_password
from utils.wordpress import (
    push_article, md_to_html, CATEGORIES, count_em_dash, slugify, strip_em_dash,
)
from utils.gold import tldr as g_tldr, faq as g_faq, sources as g_sources, assemble
from utils import infographies as ig

st.set_page_config(page_title="Publication WordPress", page_icon="🚀", layout="wide")
check_password()

st.title("🚀 Publication WordPress")
st.caption("Pousse un article mis en forme (gold standard) sur conso-energie.fr. Statut brouillon par défaut.")

WP_URL = st.secrets.get("WP_INGEST_URL", "")
WP_TOKEN = st.secrets.get("WP_INGEST_TOKEN", "")
if not WP_URL or not WP_TOKEN:
    st.error("Secrets manquants : ajoute WP_INGEST_URL et WP_INGEST_TOKEN dans les paramètres de l'app.")
    st.stop()


# ---------- Pré-remplissage depuis la page Rédaction ----------
def _clean_title(raw: str) -> str:
    if not raw:
        return ""
    # title_result / meta_result peuvent contenir des préfixes type "Title : ...".
    line = raw.strip().splitlines()[0]
    return strip_em_dash(line.split(":", 1)[-1].strip() if line.lower().startswith(("title", "titre", "meta")) else line)


pre_md = st.session_state.get("article_content") or ""
pre_title = _clean_title(st.session_state.get("title_result", "")) or _clean_title(st.session_state.get("target_keyword", ""))
pre_meta = _clean_title(st.session_state.get("meta_result", ""))

if pre_md:
    st.success("Article récupéré depuis la page Rédaction Contenu.")
else:
    st.info("Aucun article en mémoire : tu peux coller du markdown ci-dessous.")


# ---------- Helpers de parsing des infographies ----------
def _lines(txt):
    return [l.strip() for l in (txt or "").splitlines() if l.strip()]


def _split(line, n):
    parts = [p.strip() for p in line.split("|")]
    parts += [""] * (n - len(parts))
    return parts[:n]


def build_infographie(kind, title, caption, raw, raw2, comp_titles):
    if kind == "Barres":
        rows = []
        for l in _lines(raw):
            label, val, disp = _split(l, 3)
            try:
                v = float(val.replace(",", "."))
            except ValueError:
                v = 0
            rows.append((label, v, disp or val))
        return ig.barres(title, rows, caption)
    if kind == "Étapes numérotées":
        return ig.etapes(title, [tuple(_split(l, 2)) for l in _lines(raw)], caption)
    if kind == "Frise (timeline)":
        return ig.timeline(title, [tuple(_split(l, 2)) for l in _lines(raw)], caption)
    if kind == "Cartes (chiffres clés)":
        return ig.cartes(title, [tuple(_split(l, 2)) for l in _lines(raw)], caption)
    if kind == "Comparatif 2 colonnes":
        lt, rt = comp_titles
        return ig.comparatif(title, lt, _lines(raw), rt, _lines(raw2), caption)
    if kind == "Échelle DPE (A-G)":
        return ig.echelle_dpe(title or "Les classes du DPE, de A à G", caption)
    return ""


# ---------- Formulaire ----------
col1, col2 = st.columns([2, 1])
with col2:
    st.subheader("Métadonnées")
    titre = st.text_input("Titre", value=pre_title)
    slug = st.text_input("Slug", value=slugify(titre) if titre else "")
    categorie = st.selectbox("Catégorie", CATEGORIES, index=3)
    extrait = st.text_area("Extrait / meta description", value=pre_meta, height=80)
    statut = st.radio("Statut", ["draft", "publish"], horizontal=True,
                      format_func=lambda s: "Brouillon" if s == "draft" else "Publier en ligne")
    overwrite = st.checkbox("Écraser si le slug existe déjà")

with col1:
    st.subheader("Contenu (markdown)")
    md = st.text_area("Corps de l'article", value=pre_md, height=320,
                      help="Markdown : ## titres, **gras**, [ancre](url), tableaux, listes.")

st.divider()

# ---------- Blocs gold ----------
with st.expander("➕ Bloc « réponse en bref » (TL;DR)", expanded=bool(pre_md)):
    tldr_txt = st.text_area("Texte de la réponse en bref", height=90,
                            placeholder="Réponse synthétique de 2 à 4 phrases, optimisée pour le featured snippet.")

with st.expander("📊 Infographie (optionnelle)"):
    kind = st.selectbox("Type d'infographie", [
        "Aucune", "Barres", "Étapes numérotées", "Frise (timeline)",
        "Cartes (chiffres clés)", "Comparatif 2 colonnes", "Échelle DPE (A-G)"])
    info_title = st.text_input("Titre de l'infographie", key="ig_title")
    info_caption = st.text_input("Légende (figcaption)", key="ig_caption")
    info_raw = info_raw2 = ""
    comp_titles = ("", "")
    if kind in ("Barres", "Étapes numérotées", "Frise (timeline)", "Cartes (chiffres clés)"):
        ex = {
            "Barres": "Combles perdus | 8.5 | R 8,5\nMurs | 4 | R 4",
            "Étapes numérotées": "Coupez le chauffage | Laissez refroidir\nOuvrez la purge | Jusqu'au filet d'eau",
            "Frise (timeline)": "G | Depuis 2025\nF | 2028\nE | 2034",
            "Cartes (chiffres clés)": "0 % | d'intérêts\n50 000 € | jusqu'à",
        }[kind]
        info_raw = st.text_area("Données (une ligne par élément, séparées par |)", value="", placeholder=ex, height=120)
    elif kind == "Comparatif 2 colonnes":
        cc1, cc2 = st.columns(2)
        with cc1:
            lt = st.text_input("Titre colonne gauche", value="Option A")
            info_raw = st.text_area("Éléments gauche (1 par ligne)", height=120)
        with cc2:
            rt = st.text_input("Titre colonne droite", value="Option B")
            info_raw2 = st.text_area("Éléments droite (1 par ligne)", height=120)
        comp_titles = (lt, rt)

with st.expander("❓ FAQ (questions fréquentes + schema)"):
    faq_raw = st.text_area("Une Q&R par ligne, au format : Question | Réponse", height=160,
                           placeholder="Combien de temps dure un DPE ? | Un DPE est valable 10 ans.")

with st.expander("📚 Sources (E-E-A-T)"):
    sources_raw = st.text_area("Une source par ligne", height=80,
                               placeholder="ADEME, isolation des combles.\nservice-public.fr, DPE.")


# ---------- Assemblage ----------
def assembler_html():
    body = md_to_html(md) if md.strip() else ""
    infographie = ""
    if kind != "Aucune":
        infographie = build_infographie(kind, info_title, info_caption, info_raw, info_raw2, comp_titles)
    # Insère l'infographie après le 1er H2 si possible, sinon en tête du corps.
    if infographie and "</h2>" in body:
        i = body.find("</h2>") + 5
        body = body[:i] + infographie + body[i:]
    elif infographie:
        body = infographie + body
    faq_items = [tuple(p.strip() for p in l.split("|", 1)) for l in _lines(faq_raw) if "|" in l]
    return assemble(
        g_tldr(tldr_txt) if tldr_txt.strip() else "",
        body,
        g_faq(faq_items) if faq_items else "",
        g_sources(_lines(sources_raw)),
    )


final_html = assembler_html()

st.divider()
cprev, cinfo = st.columns([3, 1])
with cinfo:
    st.metric("Mots (corps)", len((md or "").split()))
    ed = count_em_dash(final_html)
    st.metric("Tirets cadratins", ed, help="Doit être 0 (nettoyés automatiquement au push).")
    st.metric("Liens internes", final_html.count('href="/') + final_html.count('href="https://www.conso-energie.fr'))

with cprev:
    st.subheader("Aperçu")
    preview_css = (
        "<style>.cep{font-family:system-ui;max-width:720px}.cep .tldr{background:#f3f3ff;border-left:4px solid #000091;"
        "padding:12px 16px;border-radius:4px}.cep h2{color:#000091}.cep .infographie{border:1px solid #dcdce6;border-radius:6px;padding:12px}"
        ".cep table{border-collapse:collapse;width:100%}.cep th{background:#000091;color:#fff;padding:6px}.cep td{border-bottom:1px solid #dcdce6;padding:6px}"
        ".cep details{border:1px solid #dcdce6;border-radius:4px;margin-bottom:6px;padding:8px}.cep .sources{font-size:.85rem;color:#525275}</style>"
    )
    st.components.v1.html(f"{preview_css}<div class='cep'>{final_html}</div>", height=520, scrolling=True)

st.divider()
if st.button("🚀 Publier sur WordPress", type="primary", disabled=not (titre and md)):
    try:
        res = push_article(
            WP_URL, WP_TOKEN, title=titre, content_html=final_html,
            category=categorie, excerpt=extrait, slug=slug, status=statut,
            metadesc=extrait, overwrite=overwrite,
        )
        verbe = "mis à jour" if res.get("updated") else "créé"
        st.success(f"Article {verbe} ({res.get('status')}) : id {res.get('id')}")
        st.markdown(f"[Voir l'article]({res.get('link')})")
    except RuntimeError as e:
        st.error(str(e))
