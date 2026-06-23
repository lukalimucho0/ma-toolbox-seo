"""
Client de publication vers WordPress (conso-energie.fr).

Pousse un article sur l'endpoint d'ingestion custom du thème
(/wp-json/conso-energie/v1/article) : le HTML est encodé en base64 pour
franchir le WAF d'o2switch, l'auth se fait par token (en-tête X-CE-Token).

Utilisation :
    from utils.wordpress import push_article, md_to_html, strip_em_dash
    res = push_article(url, token, title="...", content_html="...",
                       category="chauffage", excerpt="...", status="draft")
"""

from __future__ import annotations

import base64
import re
import time
import unicodedata
import requests

# Catégories disponibles côté WordPress (slugs).
CATEGORIES = [
    "film-solaire", "climatisation", "isolation", "chauffage",
    "chauffe-eau", "diagnostics", "aides",
]


def slugify(text: str) -> str:
    """Slug SEO simple, sans accents."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return re.sub(r"-{2,}", "-", text)[:80]


def strip_em_dash(text: str) -> str:
    """Supprime les tirets cadratins/demi-cadratins (règle stricte du projet)."""
    text = text.replace(" — ", " : ").replace(" – ", " : ")
    text = text.replace("—", "-").replace("–", "-")
    return text


def count_em_dash(text: str) -> int:
    return text.count("—") + text.count("–")


def dedupe_md_links(md: str) -> str:
    """1 URL = 1 lien : garde la première occurrence de chaque lien markdown,
    délie les suivantes (remplace [ancre](url) par l'ancre seule)."""
    seen = set()

    def repl(m):
        text, url = m.group(1), m.group(2)
        if url in seen:
            return text
        seen.add(url)
        return m.group(0)

    return re.sub(r"\[([^\]]+)\]\((\S+?)\)", repl, md or "")


def md_to_html(md: str) -> str:
    """Convertit le markdown de l'outil de rédaction en HTML (compatible .prose).

    Nécessite le paquet `markdown` (ajouté au requirements). Repli minimal si absent.
    """
    md = strip_em_dash(md)
    try:
        import markdown as _md
        html = _md.markdown(
            md,
            extensions=["tables", "sane_lists", "fenced_code", "nl2br"],
            output_format="html5",
        )
        return html
    except Exception:
        # Repli très basique (titres, gras, liens, paragraphes).
        out = []
        for block in md.split("\n\n"):
            b = block.strip()
            if not b:
                continue
            if b.startswith("### "):
                out.append(f"<h3>{b[4:].strip()}</h3>")
            elif b.startswith("## "):
                out.append(f"<h2>{b[3:].strip()}</h2>")
            else:
                b = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", b)
                b = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', b)
                out.append(f"<p>{b}</p>")
        return "\n".join(out)


def push_article(
    url: str,
    token: str,
    title: str,
    content_html: str,
    category: str = "",
    excerpt: str = "",
    slug: str = "",
    status: str = "draft",
    metadesc: str = "",
    overwrite: bool = False,
    post_type: str = "post",
    timeout: int = 30,
) -> dict:
    """Pousse l'article. Retourne un dict {ok, id, slug, status, link, ...}.

    Lève une RuntimeError avec un message clair en cas d'échec.
    """
    if not url or not token:
        raise RuntimeError("URL ou token d'ingestion manquant (vérifie les secrets Streamlit).")
    if not title.strip():
        raise RuntimeError("Le titre est obligatoire.")
    if not content_html.strip():
        raise RuntimeError("Le contenu est vide.")

    content_html = strip_em_dash(content_html)
    payload = {
        "title": strip_em_dash(title),
        "slug": slug or slugify(title),
        "category": category,
        "excerpt": strip_em_dash(excerpt),
        "content_b64": base64.b64encode(content_html.encode("utf-8")).decode("ascii"),
        "status": status,
        "metadesc": strip_em_dash(metadesc),
        "overwrite": bool(overwrite),
        "post_type": post_type,
    }
    headers = {
        "X-CE-Token": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
        # UA neutre : le WAF o2switch bloque le User-Agent par défaut de requests.
        "User-Agent": "Mozilla/5.0 (compatible; ConsoEnergieToolbox/1.0)",
    }
    # Retry sur erreurs serveur transitoires de l'hébergement mutualisé (502/503/504, timeouts).
    r = None
    last_err = ""
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_err = str(e)
            r = None
        if r is not None and r.status_code not in (502, 503, 504):
            break
        last_err = last_err or (f"HTTP {r.status_code} (serveur momentanément indisponible)" if r is not None else last_err)
        if attempt < 2:
            time.sleep(3 * (attempt + 1))
    if r is None:
        raise RuntimeError(f"Connexion impossible à WordPress après plusieurs tentatives : {last_err}")
    if r.status_code in (502, 503, 504):
        raise RuntimeError(
            "Le serveur WordPress est momentanément indisponible (erreur " + str(r.status_code) +
            "). C'est transitoire sur l'hébergement mutualisé : réessaie dans quelques instants."
        )

    try:
        body = r.json()
    except ValueError:
        body = {}

    if r.status_code in (200, 201) and body.get("ok"):
        return body

    if r.status_code == 409:
        raise RuntimeError(
            "Un article existe déjà avec ce slug. Coche « Écraser » pour le mettre à jour "
            f"(id existant : {body.get('id')})."
        )
    if r.status_code == 401:
        raise RuntimeError("Token refusé (401). Vérifie WP_INGEST_TOKEN dans les secrets.")
    msg = body.get("message") or r.text[:300] or f"HTTP {r.status_code}"
    raise RuntimeError(f"Échec de publication ({r.status_code}) : {msg}")
