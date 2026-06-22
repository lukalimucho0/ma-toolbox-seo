"""
Générateurs d'infographies SVG aux couleurs de la charte conso-energie.fr.

Chaque fonction renvoie un bloc HTML <figure class="infographie">…</figure> prêt
à être inséré dans le contenu d'un article. Le SVG est produit sur UNE seule ligne
(aucun retour à la ligne interne) pour éviter que wpautop n'y insère des <br>.

Palette : bleu #000091, bleu moyen #3a3ab0, bleu clair #8a8ad0, rouge #e1000f,
vert #18a05a, encre #1b1b2f, gris #525275.
"""

from __future__ import annotations
from html import escape

C_BLUE = "#000091"
C_BLUE2 = "#3a3ab0"
C_BLUE3 = "#6a6ac4"
C_BLUE4 = "#8a8ad0"
C_RED = "#e1000f"
C_GREEN = "#18a05a"
C_INK = "#1b1b2f"
C_SOFT = "#525275"
FONT = 'font-family="Figtree, system-ui, sans-serif"'


def _figure(svg: str, caption: str = "") -> str:
    cap = f"<figcaption>{escape(caption)}</figcaption>" if caption else ""
    return f'<figure class="infographie">{svg}{cap}</figure>'


def _t(x, y, s, size=12, color=C_INK, weight=None, anchor="start"):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{w}>{escape(str(s))}</text>'


def barres(title: str, rows: list, caption: str = "", note: str = "") -> str:
    """Barres horizontales. rows = [(label, value:float, display:str), ...]."""
    rows = rows[:8]
    maxv = max((float(v) for _, v, _ in rows), default=1) or 1
    x0, xmax = 210, 690
    span = xmax - x0
    top = 46
    rh = 36
    h = top + len(rows) * rh + (34 if note else 18)
    parts = [f'<svg viewBox="0 0 720 {h}" role="img" aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg" {FONT}>']
    parts.append(_t(0, 22, title, 17, C_BLUE, "700"))
    for i, (label, value, disp) in enumerate(rows):
        y = top + i * rh
        w = max(6, int(span * float(value) / maxv))
        col = C_BLUE if float(value) >= maxv else (C_BLUE2 if float(value) >= maxv * 0.55 else C_BLUE4)
        parts.append(f'<rect x="{x0}" y="{y}" width="{w}" height="24" rx="3" fill="{col}"/>')
        parts.append(_t(x0 - 10, y + 17, label, 12, C_INK, anchor="end"))
        parts.append(_t(x0 + w + 8, y + 17, disp, 12, C_BLUE, "700"))
    if note:
        parts.append(_t(0, h - 10, note, 11, C_SOFT))
    parts.append("</svg>")
    return _figure("".join(parts), caption)


def etapes(title: str, steps: list, caption: str = "") -> str:
    """Étapes numérotées verticales. steps = [(titre, sous-texte), ...] ou [str, ...]."""
    steps = steps[:7]
    top = 46
    rh = 50
    h = top + len(steps) * rh
    parts = [f'<svg viewBox="0 0 720 {h}" role="img" aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg" {FONT}>']
    parts.append(_t(0, 22, title, 17, C_BLUE, "700"))
    for i, step in enumerate(steps):
        t1, t2 = (step if isinstance(step, (list, tuple)) else (step, ""))
        cy = top + i * rh + 6
        col = C_GREEN if i == len(steps) - 1 else C_BLUE
        parts.append(f'<circle cx="30" cy="{cy}" r="16" fill="{col}"/>')
        parts.append(_t(30, cy + 5, i + 1, 13, "#fff", "700", "middle"))
        parts.append(_t(58, cy - 2, t1, 13.5, C_INK))
        if t2:
            parts.append(_t(58, cy + 16, t2, 11.5, C_SOFT))
    parts.append("</svg>")
    return _figure("".join(parts), caption)


def timeline(title: str, nodes: list, caption: str = "", colors: list = None) -> str:
    """Frise horizontale. nodes = [(label, sous-label), ...]."""
    nodes = nodes[:5]
    n = len(nodes)
    parts = [f'<svg viewBox="0 0 720 210" role="img" aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg" {FONT}>']
    parts.append(_t(0, 22, title, 17, C_BLUE, "700"))
    xs = [120 + i * (510 // max(1, n - 1)) for i in range(n)] if n > 1 else [360]
    parts.append(f'<line x1="{xs[0]}" y1="110" x2="{xs[-1]}" y2="110" stroke="#dcdce6" stroke-width="3"/>')
    pal = colors or [C_RED, "#ea6a25", "#f1a12a", C_BLUE2, C_BLUE]
    for i, (label, sub) in enumerate(nodes):
        x = xs[i]
        col = pal[i % len(pal)]
        parts.append(f'<circle cx="{x}" cy="110" r="26" fill="{col}"/>')
        parts.append(_t(x, 117, label, 15, "#fff", "700", "middle"))
        parts.append(_t(x, 160, sub, 12.5, C_INK, "700", "middle"))
    parts.append("</svg>")
    return _figure("".join(parts), caption)


def comparatif(title: str, left_title: str, left_items: list, right_title: str, right_items: list, caption: str = "") -> str:
    """Deux colonnes (gauche bleu, droite vert)."""
    li, ri = left_items[:6], right_items[:6]
    rows = max(len(li), len(ri))
    h = 60 + rows * 24 + 20
    bh = h - 44
    parts = [f'<svg viewBox="0 0 720 {h}" role="img" aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg" {FONT}>']
    parts.append(_t(0, 22, title, 17, C_BLUE, "700"))
    parts.append(f'<rect x="14" y="40" width="340" height="{bh}" rx="8" fill="#e7e7ff" stroke="{C_BLUE}" stroke-width="1.5"/>')
    parts.append(_t(184, 66, left_title, 15, C_BLUE, "700", "middle"))
    for i, it in enumerate(li):
        parts.append(_t(34, 96 + i * 24, "• " + it, 12.5, C_INK))
    parts.append(f'<rect x="366" y="40" width="340" height="{bh}" rx="8" fill="#eafaf1" stroke="{C_GREEN}" stroke-width="1.5"/>')
    parts.append(_t(536, 66, right_title, 15, C_GREEN, "700", "middle"))
    for i, it in enumerate(ri):
        parts.append(_t(386, 96 + i * 24, "• " + it, 12.5, C_INK))
    parts.append("</svg>")
    return _figure("".join(parts), caption)


def cartes(title: str, cards: list, caption: str = "") -> str:
    """Cartes de chiffres clés. cards = [(grand, petit), ...] (max 4)."""
    cards = cards[:4]
    n = len(cards)
    gap = 12
    total = 692
    w = (total - (n - 1) * gap) // n
    parts = [f'<svg viewBox="0 0 720 180" role="img" aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg" {FONT}>']
    parts.append(_t(0, 22, title, 17, C_BLUE, "700"))
    for i, (big, small) in enumerate(cards):
        x = 14 + i * (w + gap)
        parts.append(f'<rect x="{x}" y="44" width="{w}" height="118" rx="8" fill="#e7e7ff" stroke="{C_BLUE}" stroke-width="1.5"/>')
        size = 26 if len(str(big)) <= 6 else 20
        parts.append(_t(x + w // 2, 100, big, size, C_BLUE, "800", "middle"))
        parts.append(_t(x + w // 2, 128, small, 12.5, C_INK, anchor="middle"))
    parts.append("</svg>")
    return _figure("".join(parts), caption)


def echelle_dpe(title: str = "Les classes du DPE, de A à G", caption: str = "") -> str:
    """Étiquette DPE A->G colorée."""
    data = [("A", "#319a4f", 130), ("B", "#5fb646", 175), ("C", "#a7c83a", 220),
            ("D", "#f4d100", 265), ("E", "#f1a12a", 310), ("F", "#ea6a25", 355), ("G", "#e30613", 400)]
    parts = [f'<svg viewBox="0 0 720 270" role="img" aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg" {FONT}>']
    parts.append(_t(0, 22, title, 17, C_BLUE, "700"))
    y = 40
    for letter, color, w in data:
        parts.append(f'<rect x="60" y="{y}" width="{w}" height="22" rx="3" fill="{color}"/>')
        parts.append(_t(40, y + 16, letter, 13, color, "700", "end"))
        y += 30
    parts.append("</svg>")
    return _figure("".join(parts), caption)
