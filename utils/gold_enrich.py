"""
Enrichissement « gold standard » d'un article rédigé.

Un appel IA transforme l'article (markdown) en éléments gold conso-energie.fr :
réponse en bref (TL;DR), FAQ (question/réponse), une infographie (type + données),
plus les métadonnées (slug, meta description, catégorie, sources).

build_gold_html() assemble ensuite le HTML final prêt à publier.
"""

from __future__ import annotations
import json
import re

from .wordpress import md_to_html, slugify, strip_em_dash, CATEGORIES, dedupe_md_links
from .gold import tldr as g_tldr, faq as g_faq, sources as g_sources, assemble
from . import infographies as ig

PROMPT = """Tu es un éditeur SEO senior pour conso-energie.fr (guide de l'énergie du logement).
À partir de l'article ci-dessous (mot-clé cible : "{kw}"), produis des éléments de mise en forme.

RÈGLES STRICTES :
- Français. JAMAIS de tiret cadratin (—) ni demi-cadratin (–).
- Réponds UNIQUEMENT par un objet JSON valide, sans texte autour, sans bloc de code.
- Base-toi exclusivement sur le contenu de l'article (n'invente pas de chiffres).

Schéma JSON attendu :
{{
  "tldr": "Réponse synthétique de 2 à 4 phrases, optimisée featured snippet.",
  "faq": [{{"q": "Question ?", "a": "Réponse courte."}}],  // 4 entrées
  "metadesc": "Meta description <= 155 caractères, accrocheuse.",
  "slug": "slug-seo-court",
  "category": "une valeur parmi : {cats}",
  "sources": ["Source 1", "Source 2"],
  "infographics": [
    {{
     "type": "barres | etapes | timeline | comparatif | cartes | echelle_dpe",
     "title": "Titre court de l'infographie",
     "caption": "Légende sous l'infographie",
     "rows": [["Label", 8.5, "R 8,5"]],          // barres : [label, valeur_num, affichage]
     "steps": [["Titre étape", "sous-texte"]],    // etapes
     "nodes": [["G", "Depuis 2025"]],             // timeline : [label, sous-label]
     "cards": [["0 %", "d'intérêts"]],            // cartes : [grand, petit]
     "left_title": "", "right_title": "", "left": [], "right": []  // comparatif
    }}
  ]
}}
Propose 0 à 2 infographies VRAIMENT pertinentes (liste vide si rien ne s'y prête), chacune sur une donnée DIFFÉRENTE de l'article et adaptée à une section distincte. N'utilise que des chiffres présents dans l'article.

ARTICLE :
{article}
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Réponse IA sans JSON exploitable.")
    return json.loads(text[start:end + 1])


def enrich_to_gold(ai_analyzer, keyword: str, article_md: str) -> dict:
    """Appelle l'IA et renvoie le dict d'enrichissement (validé/nettoyé)."""
    prompt = PROMPT.format(kw=keyword, cats=", ".join(CATEGORIES), article=article_md[:12000])
    raw = ai_analyzer.analyze_with_custom_prompt(prompt, max_tokens=2000, temperature=0.4)
    data = _parse_json(raw)
    data.setdefault("tldr", "")
    data.setdefault("faq", [])
    data.setdefault("sources", [])
    if not isinstance(data.get("infographics"), list):
        one = data.get("infographic")
        data["infographics"] = [one] if isinstance(one, dict) else []
    if data.get("category") not in CATEGORIES:
        data["category"] = ""
    if not data.get("slug"):
        data["slug"] = slugify(keyword)
    return data


def _build_infographic(spec: dict) -> str:
    if not spec or spec.get("type") in (None, "", "aucune"):
        return ""
    t = spec.get("type")
    title = spec.get("title", "")
    cap = spec.get("caption", "")
    try:
        if t == "barres":
            rows = [(r[0], float(r[1]), (r[2] if len(r) > 2 else str(r[1]))) for r in spec.get("rows", []) if r]
            return ig.barres(title, rows, cap)
        if t == "etapes":
            return ig.etapes(title, [tuple(s) if isinstance(s, (list, tuple)) else (s, "") for s in spec.get("steps", [])], cap)
        if t == "timeline":
            return ig.timeline(title, [tuple(n) for n in spec.get("nodes", []) if n], cap)
        if t == "cartes":
            return ig.cartes(title, [tuple(c) for c in spec.get("cards", []) if c], cap)
        if t == "comparatif":
            return ig.comparatif(title, spec.get("left_title", ""), spec.get("left", []),
                                 spec.get("right_title", ""), spec.get("right", []), cap)
        if t == "echelle_dpe":
            return ig.echelle_dpe(title or "Les classes du DPE, de A à G", cap)
    except Exception:
        return ""
    return ""


def _insert_infographics(body: str, figs: list) -> str:
    """Insère chaque infographie après un H2 distinct (fig 1 après le 1er H2, etc.)."""
    positions = [m.end() for m in re.finditer(r"</h2>", body)]
    if not positions:
        return "".join(figs) + body
    n = min(len(figs), len(positions))
    out = body
    for pos, fig in sorted(zip(positions[:n], figs[:n]), key=lambda x: -x[0]):
        out = out[:pos] + fig + out[pos:]
    if len(figs) > n:  # surplus éventuel en fin d'article
        out += "".join(figs[n:])
    return out


def build_gold_html(article_md: str, enrich: dict) -> str:
    """Assemble le HTML gold final (TL;DR + corps + infographie + FAQ + sources)."""
    article_md = dedupe_md_links(article_md or "")
    body = md_to_html(article_md) if article_md.strip() else ""
    # Le H1 est déjà rendu par le template d'article : on le retire du contenu.
    body = re.sub(r"(?is)<h1\b[^>]*>.*?</h1>\s*", "", body)
    specs = enrich.get("infographics")
    if not isinstance(specs, list):
        one = enrich.get("infographic")
        specs = [one] if isinstance(one, dict) else []
    figs = [h for h in (_build_infographic(s) for s in specs) if h]
    if figs:
        body = _insert_infographics(body, figs)
    faq_items = [(f.get("q", ""), f.get("a", "")) for f in enrich.get("faq", []) if isinstance(f, dict)]
    return assemble(
        g_tldr(enrich["tldr"]) if enrich.get("tldr") else "",
        body,
        g_faq(faq_items) if faq_items else "",
        g_sources(enrich.get("sources", [])),
    )
