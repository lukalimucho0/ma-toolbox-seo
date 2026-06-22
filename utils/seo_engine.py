"""
Passerelle vers le moteur SEO de la page « Rédaction Contenu ».

Le moteur (DataForSEO, extraction concurrents, structure Hn, prompts, rédaction)
vit dans pages/1_✍️_Rédaction_Contenu.py et est quasi sans Streamlit. Plutôt que
de le dupliquer, on charge ce fichier comme un module en neutralisant son UI :
on pose un drapeau sys._CE_ENGINE_IMPORT, et la page ne lance main() que si ce
drapeau est absent (exécution normale par Streamlit).

Expose : SEOBriefGenerator, PromptTemplates, HeadingParser, run_writing_engine,
DataForSEOConfig, AIAnalyzer, InternalLinksParser, CompetitorData, HeadingNode.
"""

from __future__ import annotations
import os
import sys
import importlib.util

_PAGE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pages", "1_✍️_Rédaction_Contenu.py")


def _load():
    sys._CE_ENGINE_IMPORT = True
    try:
        spec = importlib.util.spec_from_file_location("ce_redaction_engine", _PAGE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys._CE_ENGINE_IMPORT = False


_mod = _load()

SEOBriefGenerator = _mod.SEOBriefGenerator
PromptTemplates = _mod.PromptTemplates
HeadingParser = _mod.HeadingParser
run_writing_engine = _mod.run_writing_engine
DataForSEOConfig = _mod.DataForSEOConfig
AIAnalyzer = _mod.AIAnalyzer
InternalLinksParser = _mod.InternalLinksParser
CompetitorData = _mod.CompetitorData
HeadingNode = _mod.HeadingNode


def nodes_to_markdown(nodes) -> str:
    """Reconstruit le markdown complet à partir des HeadingNode rédigés."""
    lvl = {1: "#", 2: "##", 3: "###", 4: "####"}
    out = []
    for n in nodes:
        out.append(f"{lvl.get(n.level, '####')} {n.text}\n")
        if getattr(n, "content", ""):
            out.append(n.content + "\n")
    return "\n".join(out)
