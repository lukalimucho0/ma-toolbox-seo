"""
Assemblage des blocs « gold standard » conso-energie.fr :
bloc réponse (TL;DR), FAQ accordéon + schema FAQPage (JSON-LD), sources.

Tous les blocs respectent les classes CSS du thème (.tldr, .faq, .faq-item,
.sources) et l'interdiction du tiret cadratin.
"""

from __future__ import annotations
import json
from html import escape
from .wordpress import strip_em_dash


def tldr(text: str, titre: str = "La réponse en bref") -> str:
    text = strip_em_dash(text).strip()
    return (
        f'<div class="tldr"><p><strong>{escape(titre)}</strong></p>'
        f'<p>{strip_em_dash(text)}</p></div>'
    )


def faq(items: list, titre: str = "Questions fréquentes") -> str:
    """items = [(question, réponse), ...]. Renvoie le bloc visible + le JSON-LD."""
    items = [(strip_em_dash(q).strip(), strip_em_dash(a).strip()) for q, a in items if q and a]
    if not items:
        return ""
    visible = [f'<h2>{escape(titre)}</h2><div class="faq">']
    for q, a in items:
        visible.append(f'<details class="faq-item"><summary>{escape(q)}</summary><p>{escape(a)}</p></details>')
    visible.append("</div>")
    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {"@type": "Question", "name": q,
             "acceptedAnswer": {"@type": "Answer", "text": a}}
            for q, a in items
        ],
    }
    ld = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    script = f'<script type="application/ld+json">{ld}</script>'
    return "".join(visible) + script


def sources(items: list) -> str:
    items = [strip_em_dash(s).strip() for s in items if s and s.strip()]
    if not items:
        return ""
    lis = "".join(f"<li>{escape(s)}</li>" for s in items)
    return f'<ul class="sources">{lis}</ul>'


def assemble(*blocks: str) -> str:
    """Concatène les blocs non vides dans l'ordre fourni."""
    return "\n".join(b for b in blocks if b and b.strip())
