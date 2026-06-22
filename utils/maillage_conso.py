"""
Table de maillage interne de conso-energie.fr.

Chaque entrée = (url, ancre) où l'ancre est le mot-clé cible de la page.
L'outil de rédaction pioche dans cette liste l'ancre pertinente selon le contenu
de chaque section et place le lien (1 URL = 1 lien, géré par dedupe_md_links).

À tenir à jour quand de nouvelles pages/articles sont publiés.
"""

MAILLAGE = [
    # Piliers & pages transversales
    ("/film-solaire/", "film solaire"),
    ("/climatisation/", "climatisation"),
    ("/isolation/", "isolation thermique"),
    ("/chauffage/", "chauffage"),
    ("/chauffe-eau/", "chauffe-eau"),
    ("/diagnostics/", "diagnostic énergétique"),
    ("/aides/", "aides à la rénovation énergétique"),
    ("/aides/maprimerenov/", "MaPrimeRénov'"),
    ("/simulateur-renovation/", "simulateur de rénovation"),
    ("/outils/", "simulateur d'énergie"),
    ("/zones/", "près de chez vous"),

    # Film solaire
    ("/film-solaire/pose-film-solaire/", "pose d'un film solaire"),
    ("/film-solaire/prix/", "prix d'un film solaire"),
    ("/film-solaire/avis/", "avis sur le film solaire"),
    ("/film-solaire/film-anti-chaleur-fenetre/", "film anti-chaleur pour fenêtre"),
    ("/film-solaire/film-anti-chaleur-velux/", "film anti-chaleur pour Velux"),
    ("/film-solaire/film-anti-chaleur-transparent/", "film anti-chaleur transparent"),
    ("/film-solaire/film-anti-chaleur-anti-froid/", "film anti-chaleur et anti-froid"),
    ("/film-solaire/film-anti-uv/", "film anti-UV"),
    ("/film-solaire/film-miroir-sans-tain/", "film miroir sans tain"),

    # Climatisation
    ("/climatisation/reversible/", "climatisation réversible"),
    ("/climatisation/gainable/", "climatisation gainable"),
    ("/climatisation/split/", "climatisation split"),
    ("/climatisation/sans-unite-exterieure/", "climatisation sans unité extérieure"),
    ("/climatisation/installation/", "installation d'une climatisation"),
    ("/climatisation/entretien/", "entretien de la climatisation"),
    ("/climatisation/prix/", "prix d'une climatisation"),

    # Isolation
    ("/isolation/exterieure/", "isolation par l'extérieur"),
    ("/isolation/combles/", "isolation des combles"),
    ("/isolation/toiture/", "isolation de la toiture"),
    ("/isolation/murs/", "isolation des murs"),
    ("/isolation/sol/", "isolation du sol"),
    ("/isolation/isolants/", "quel isolant choisir"),
    ("/isolation/prix/", "prix de l'isolation"),

    # Chauffage
    ("/chauffage/pompe-a-chaleur/", "pompe à chaleur"),
    ("/chauffage/poele-granules/", "poêle à granulés"),
    ("/chauffage/poele-bois/", "poêle à bois"),
    ("/chauffage/chaudiere-gaz/", "chaudière gaz"),
    ("/chauffage/radiateur-electrique/", "radiateur électrique"),
    ("/chauffage/plancher-chauffant/", "plancher chauffant"),
    ("/chauffage/entretien/", "entretien du chauffage"),
    ("/chauffage/prix/", "prix du chauffage"),

    # Chauffe-eau
    ("/chauffe-eau/thermodynamique/", "chauffe-eau thermodynamique"),
    ("/chauffe-eau/solaire/", "chauffe-eau solaire"),
    ("/chauffe-eau/electrique/", "chauffe-eau électrique"),
    ("/chauffe-eau/gaz/", "chauffe-eau gaz"),
    ("/chauffe-eau/instantane/", "chauffe-eau instantané"),
    ("/chauffe-eau/installation/", "installation d'un chauffe-eau"),
    ("/chauffe-eau/entretien/", "entretien d'un chauffe-eau"),
    ("/chauffe-eau/pannes/", "pannes de chauffe-eau"),
    ("/chauffe-eau/prix/", "prix d'un chauffe-eau"),

    # Articles de blog
    ("/actualites/film-solaire/film-solaire-interieur-ou-exterieur/", "film solaire intérieur ou extérieur"),
    ("/actualites/film-solaire/film-solaire-autocollant-ou-electrostatique/", "film solaire autocollant ou électrostatique"),
    ("/actualites/film-solaire/film-solaire-combien-de-degres/", "gain de température d'un film solaire"),
    ("/actualites/climatisation/comment-fonctionne-une-climatisation/", "comment fonctionne une climatisation"),
    ("/actualites/climatisation/comment-fonctionne-climatisation-reversible/", "fonctionnement d'une climatisation réversible"),
    ("/actualites/climatisation/clim-mode-dry-ou-cool/", "mode dry ou cool de la climatisation"),
    ("/actualites/climatisation/rafraichir-maison-sans-climatisation/", "rafraîchir sa maison sans climatisation"),
    ("/actualites/climatisation/bien-utiliser-climatisation-canicule/", "bien utiliser sa climatisation en canicule"),
    ("/actualites/climatisation/eteindre-climatisation-absence/", "éteindre sa climatisation en cas d'absence"),
    ("/actualites/isolation/quel-r-pour-une-bonne-isolation/", "résistance thermique R en isolation"),
    ("/actualites/isolation/meilleur-isolant-thermique/", "meilleur isolant thermique"),
    ("/actualites/isolation/meilleur-isolant-phonique/", "meilleur isolant phonique"),
    ("/actualites/isolation/electricite-avant-ou-apres-isolation/", "électricité avant ou après l'isolation"),
    ("/actualites/chauffage/comment-purger-un-radiateur/", "purger un radiateur"),
    ("/actualites/chauffage/purger-radiateur-fonte/", "purger un radiateur en fonte"),
    ("/actualites/chauffage/purger-radiateur-en-marche/", "purger un radiateur en marche"),
    ("/actualites/chauffage/quand-allumer-le-chauffage/", "quand allumer le chauffage"),
    ("/actualites/chauffe-eau/comment-vidanger-un-chauffe-eau/", "vidanger un chauffe-eau"),
    ("/actualites/chauffe-eau/temps-vidange-chauffe-eau/", "temps de vidange d'un chauffe-eau"),
    ("/actualites/chauffe-eau/chauffe-eau-qui-goutte-groupe-de-securite/", "chauffe-eau qui goutte"),
    ("/actualites/diagnostics/comment-se-passe-diagnostic-dpe/", "déroulé d'un diagnostic DPE"),
    ("/actualites/diagnostics/duree-validite-dpe/", "durée de validité d'un DPE"),
    ("/actualites/diagnostics/dpe-obligatoire-location/", "DPE obligatoire pour une location"),
    ("/actualites/aides/prime-cee-definition/", "prime CEE"),
    ("/actualites/aides/eco-ptz-definition/", "éco-PTZ"),
    ("/actualites/aides/travaux-renovation-energetique-eligibles-aides/", "travaux éligibles aux aides"),
]


def maillage_block(exclude_url: str = "") -> str:
    """Liste 'url | ancre' (une par ligne), pour alimenter le moteur de rédaction."""
    return "\n".join(f"{u} | {a}" for u, a in MAILLAGE if u != exclude_url)
