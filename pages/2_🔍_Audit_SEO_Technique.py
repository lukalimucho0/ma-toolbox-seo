"""
🔍 SEO TECHNICAL AUDIT TOOL - Expert Edition
=============================================
Outil d'audit SEO technique avancé combinant données GSC et Screaming Frog
avec analyse IA experte via Claude API.

Auteur: Expert SEO Tool
Version: 4.0 - Avec analyse GSC Indexation (404 & Redirections)

Nouveautés v4:
- Import des fichiers GSC Indexation (404 et Redirections)
- Analyse croisée 404 GSC x Screaming Frog
- Analyse croisée Redirections GSC x Screaming Frog
- Détection des 404 avec liens internes
- Détection des chaînes de redirection cassées
- Rapport IA enrichi avec données d'indexation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
from datetime import datetime
import io
import re
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Audit SEO Technique | Ma Toolbox SEO",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look pro
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
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .critical-box {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #F3F4F6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def clean_url(url: str) -> str:
    """Nettoie et normalise une URL."""
    if pd.isna(url):
        return ""
    url = str(url).strip().lower()
    url = url.rstrip('/')
    return url

def extract_path(url: str) -> str:
    """Extrait le path d'une URL."""
    if pd.isna(url) or not url:
        return ""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.path
    except:
        return url

def calculate_crawl_depth(url: str) -> int:
    """Calcule la profondeur de crawl d'une URL."""
    if pd.isna(url) or not url:
        return 0
    path = extract_path(url)
    depth = len([x for x in path.split('/') if x])
    return depth

def analyze_title_length(length) -> str:
    """Analyse la longueur d'un title."""
    try:
        length = int(length) if pd.notna(length) else 0
        if length == 0:
            return "❌ Manquant"
        elif length < 30:
            return "⚠️ Trop court (<30)"
        elif length <= 60:
            return "✅ Optimal (30-60)"
        elif length <= 70:
            return "⚠️ Limite (60-70)"
        else:
            return "❌ Trop long (>70)"
    except:
        return "❓ Inconnu"

def analyze_meta_desc_length(length) -> str:
    """Analyse la longueur d'une meta description."""
    try:
        length = int(length) if pd.notna(length) else 0
        if length == 0:
            return "❌ Manquante"
        elif length < 70:
            return "⚠️ Trop courte (<70)"
        elif length <= 155:
            return "✅ Optimale (70-155)"
        elif length <= 160:
            return "⚠️ Limite (155-160)"
        else:
            return "❌ Trop longue (>160)"
    except:
        return "❓ Inconnu"

def analyze_word_count(count) -> str:
    """Analyse le nombre de mots d'une page."""
    try:
        count = int(count) if pd.notna(count) else 0
        if count == 0:
            return "❌ Vide"
        elif count < 100:
            return "🔴 Très thin (<100)"
        elif count < 300:
            return "🟠 Thin (100-300)"
        elif count < 600:
            return "🟡 Léger (300-600)"
        elif count <= 1500:
            return "✅ Optimal (600-1500)"
        elif count <= 3000:
            return "✅ Riche (1500-3000)"
        else:
            return "📚 Très riche (>3000)"
    except:
        return "❓ Inconnu"

def format_number(n) -> str:
    """Formate un nombre avec séparateurs de milliers."""
    try:
        return f"{int(n):,}".replace(",", " ")
    except:
        return str(n)


def find_column_flexible(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Trouve une colonne dans le DataFrame parmi une liste de noms possibles.
    Recherche exacte puis partielle, insensible à la casse.
    """
    # Créer un mapping des colonnes en minuscules
    columns_list = list(df.columns)
    columns_lower = [str(c).lower().strip() for c in columns_list]
    
    # 1. Recherche exacte (insensible à la casse)
    for name in possible_names:
        name_lower = name.lower().strip()
        for i, col_lower in enumerate(columns_lower):
            if col_lower == name_lower:
                return columns_list[i]
    
    # 2. Recherche partielle (le nom recherché est contenu dans la colonne)
    for name in possible_names:
        name_lower = name.lower().strip()
        for i, col_lower in enumerate(columns_lower):
            if name_lower in col_lower:
                return columns_list[i]
    
    # 3. Recherche partielle inverse (la colonne est contenue dans le nom recherché)
    for name in possible_names:
        name_lower = name.lower().strip()
        for i, col_lower in enumerate(columns_lower):
            if col_lower in name_lower and len(col_lower) > 2:
                return columns_list[i]
    
    return None


def standardize_gsc_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Standardise les colonnes du fichier GSC.
    Retourne le DataFrame, le mapping effectué, et les erreurs éventuelles.
    """
    df = df.copy()
    mapping = {}
    errors = []
    
    # Définition des colonnes à rechercher
    column_definitions = {
        'url': ['Page', 'Pages', 'URL', 'url', 'Top pages', 'Landing Page', 'Landing page', 
                'Adresse', 'Address', 'Query', 'page', 'landing page', 'top page'],
        'clicks': ['Clics', 'Clicks', 'clicks', 'clics', 'Clic', 'Click', 'clic', 'click'],
        'impressions': ['Impressions', 'impressions', 'Impression', 'impression', 'Impr', 'impr'],
        'ctr': ['CTR', 'ctr', 'Taux de clics', 'Click Through Rate', 'Click-through rate', 
                'taux de clic', 'click through rate'],
        'position': ['Position', 'position', 'Position moyenne', 'Average position', 
                    'Avg Position', 'Avg. Position', 'Average Position', 'Rang', 'Rank',
                    'position moyenne', 'avg position']
    }
    
    # Chercher chaque colonne
    for target_name, possible_names in column_definitions.items():
        found_col = find_column_flexible(df, possible_names)
        if found_col:
            if found_col != target_name:
                df.rename(columns={found_col: target_name}, inplace=True)
            mapping[target_name] = found_col
    
    # Vérifier si on a la colonne URL
    if 'url' not in df.columns:
        # Vérifier si c'est peut-être un export par date
        if 'Date' in df.columns or 'date' in df.columns:
            errors.append("⚠️ Ce fichier semble être un export GSC par DATE et non par PAGES.")
            errors.append("👉 Dans Google Search Console, allez dans Performances > Onglet 'Pages' > Exporter")
        else:
            errors.append(f"❌ Colonne URL non trouvée. Colonnes disponibles: {list(df.columns)}")
    
    return df, mapping, errors


def standardize_sf_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Standardise les colonnes du fichier Screaming Frog.
    Retourne le DataFrame, le mapping effectué, et les erreurs éventuelles.
    """
    df = df.copy()
    mapping = {}
    errors = []
    
    # Définition des colonnes à rechercher (FR + EN)
    column_definitions = {
        'url': ['Adresse', 'Address', 'URL', 'url', 'adresse', 'address', 'Uri', 'URI', 'Page'],
        'status_code': ['Status Code', 'status_code', 'Code de statut', 'Code HTTP', 'StatusCode',
                       'code de statut', 'code http', 'status code', 'Statut'],
        'indexability': ['Indexability', 'Indexabilité', 'indexability', 'indexabilité', 'Indexable'],
        'indexability_status': ['Indexability Status', 'Statut d\'indexabilité', 'indexability_status',
                               "Statut d'indexabilité", 'statut indexabilité'],
        'title': ['Title 1', 'Title', 'Titre', 'title', 'titre', 'Meta Title', 'Title1'],
        'title_length': ['Title 1 Length', 'Longueur du Title 1', 'title_length', 'Title Length',
                        'longueur du title', 'Longueur Title', 'longueur title 1'],
        'meta_description': ['Meta Description 1', 'Meta Description', 'meta_description',
                            'Description', 'Méta description', 'méta description 1'],
        'meta_desc_length': ['Meta Description 1 Length', 'Longueur de la Meta Description 1',
                            'meta_desc_length', 'Meta Description Length', 'Longueur Meta Description',
                            'longueur de la meta description 1', 'longueur meta description'],
        'h1': ['H1-1', 'H1', 'h1', 'Heading 1', 'H1 1', 'h1-1'],
        'h1_length': ['H1-1 Length', 'Longueur du H1-1', 'h1_length', 'H1 Length', 'longueur du h1-1',
                     'Longueur H1'],
        'word_count': ['Word Count', 'Nombre de mots', 'word_count', 'Words', 'Mots',
                      'nombre de mots', 'wordcount', 'Content Length'],
        'crawl_depth': ['Crawl Depth', 'Profondeur du dossier', 'crawl_depth', 'Depth', 'Level',
                       'Profondeur', 'profondeur du dossier', 'profondeur'],
        'inlinks': ['Inlinks', 'Liens entrants', 'inlinks', 'liens entrants', 'Internal Inlinks',
                   'Liens entrants uniques'],
        'unique_inlinks': ['Unique Inlinks', 'Liens entrants uniques', 'unique_inlinks',
                          'liens entrants uniques'],
        'outlinks': ['Outlinks', 'Liens sortants', 'outlinks', 'liens sortants', 'Internal Outlinks'],
        'unique_outlinks': ['Unique Outlinks', 'Liens sortants uniques', 'unique_outlinks',
                           'liens sortants uniques'],
        'canonical': ['Canonical Link Element 1', 'canonical', 'Canonique', 
                     'Élément de lien en version canonique 1', 'Canonical URL', 'Canonical'],
        'response_time': ['Response Time', 'Temps de réponse', 'response_time', 'TTFB',
                         'temps de réponse', 'Time to First Byte']
    }
    
    # Chercher chaque colonne
    for target_name, possible_names in column_definitions.items():
        found_col = find_column_flexible(df, possible_names)
        if found_col:
            if found_col != target_name:
                df.rename(columns={found_col: target_name}, inplace=True)
            mapping[target_name] = found_col
    
    # Vérifier si on a la colonne URL
    if 'url' not in df.columns:
        errors.append(f"❌ Colonne URL/Adresse non trouvée.")
        errors.append(f"Colonnes disponibles: {list(df.columns)[:10]}...")
    
    return df, mapping, errors


def standardize_gsc_indexation_columns(df: pd.DataFrame, file_type: str) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Standardise les colonnes des fichiers d'indexation GSC (404 et redirections).
    
    Args:
        df: DataFrame à standardiser
        file_type: Type de fichier ('404' ou 'redirections')
    
    Returns:
        Tuple contenant le DataFrame standardisé, le mapping effectué, et les erreurs
    """
    df = df.copy()
    mapping = {}
    errors = []
    
    # Colonnes possibles pour les fichiers d'indexation GSC (FR + EN)
    column_definitions = {
        'url': ['URL', 'url', 'Page', 'page', 'Adresse', 'Address', 'URI', 
                'URL inspectée', 'Inspected URL', 'URL de la page', 'Page URL',
                'URL inspecte', 'Url'],
        'last_crawl': ['Dernière exploration', 'Last crawled', 'Last Crawl', 
                      'Date d\'exploration', 'Crawl Date', 'Date exploration',
                      'Dernière analyse', 'Date', 'Last crawl date',
                      'dernière exploration', 'Dernier crawl'],
        'status': ['État', 'Status', 'Statut', 'Reason', 'Raison', 
                  'Coverage status', 'État de couverture', 'Etat',
                  'état', 'statut', 'État de la page'],
        'discovery': ['Découverte', 'Discovery', 'Discovered', 'Source',
                     'Comment Google a découvert cette URL', 'How Google discovered',
                     'Découverte via', 'Discovered via', 'découverte'],
        'sitemap': ['Sitemap', 'Dans le sitemap', 'In sitemap', 'Sitemaps',
                   'sitemap', 'Sitemap URL'],
        'referring_page': ['Page de provenance', 'Referring page', 'Source page',
                          'Page référente', 'Linked from', 'Référent',
                          'page de provenance', 'Page source']
    }
    
    # Chercher chaque colonne
    for target_name, possible_names in column_definitions.items():
        found_col = find_column_flexible(df, possible_names)
        if found_col:
            if found_col != target_name:
                df.rename(columns={found_col: target_name}, inplace=True)
            mapping[target_name] = found_col
    
    # Si URL non trouvée, essayer de trouver n'importe quelle colonne contenant des URLs
    if 'url' not in df.columns:
        for col in df.columns:
            sample = df[col].dropna().head(10).tolist()
            if any('http' in str(s).lower() for s in sample):
                df.rename(columns={col: 'url'}, inplace=True)
                mapping['url'] = col
                break
    
    # Vérification finale
    if 'url' not in df.columns:
        errors.append(f"❌ Colonne URL non trouvée dans le fichier {file_type}.")
        errors.append(f"Colonnes disponibles: {list(df.columns)[:10]}...")
    
    return df, mapping, errors


# =============================================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# =============================================================================

@st.cache_data
def load_gsc_data(file) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Charge et prépare les données GSC."""
    errors = []
    
    try:
        # Lecture du fichier
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            content = file.read()
            file.seek(0)
            
            # Détecter le délimiteur
            first_lines = content.decode('utf-8', errors='ignore')[:2000]
            if '\t' in first_lines and first_lines.count('\t') > first_lines.count(','):
                delimiter = '\t'
            elif ';' in first_lines and first_lines.count(';') > first_lines.count(','):
                delimiter = ';'
            else:
                delimiter = ','
            
            try:
                df = pd.read_csv(file, encoding='utf-8', sep=delimiter)
            except:
                file.seek(0)
                df = pd.read_csv(file, encoding='latin-1', sep=delimiter)
        
        # Supprimer les lignes complètement vides
        df = df.dropna(how='all')
        
        # Standardiser les colonnes
        df, mapping, std_errors = standardize_gsc_columns(df)
        errors.extend(std_errors)
        
        if errors:
            return df, mapping, errors
        
        # Nettoyage des URLs
        if 'url' in df.columns:
            df['url'] = df['url'].apply(clean_url)
        
        # Conversion des colonnes numériques
        numeric_cols = ['clicks', 'impressions', 'position']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Traitement du CTR
        if 'ctr' in df.columns:
            if df['ctr'].dtype == object:
                df['ctr'] = df['ctr'].astype(str).str.replace('%', '').str.replace(',', '.').str.strip()
                df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)
                if df['ctr'].max() > 1:
                    df['ctr'] = df['ctr'] / 100
            elif df['ctr'].max() > 1:
                df['ctr'] = df['ctr'] / 100
        elif 'clicks' in df.columns and 'impressions' in df.columns:
            df['ctr'] = df.apply(
                lambda x: x['clicks'] / x['impressions'] if x['impressions'] > 0 else 0, 
                axis=1
            )
        
        return df, mapping, errors
    
    except Exception as e:
        errors.append(f"Erreur lors du chargement: {str(e)}")
        return pd.DataFrame(), {}, errors


@st.cache_data
def load_sf_data(file) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Charge et prépare les données Screaming Frog."""
    errors = []
    
    try:
        # Lecture du fichier
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            content = file.read()
            file.seek(0)
            
            # Détecter le délimiteur
            first_lines = content.decode('utf-8', errors='ignore')[:2000]
            if '\t' in first_lines and first_lines.count('\t') > first_lines.count(','):
                delimiter = '\t'
            elif ';' in first_lines and first_lines.count(';') > first_lines.count(','):
                delimiter = ';'
            else:
                delimiter = ','
            
            try:
                df = pd.read_csv(file, encoding='utf-8', sep=delimiter)
            except:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding='latin-1', sep=delimiter)
                except:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='utf-8')
        
        # Si la première colonne semble être un header SF, la sauter
        if len(df.columns) > 0:
            first_col = str(df.columns[0]).lower()
            if 'internal' in first_col or 'screaming' in first_col or 'crawl' in first_col:
                file.seek(0)
                if file.name.endswith('.xlsx'):
                    df = pd.read_excel(file, skiprows=1)
                else:
                    df = pd.read_csv(file, skiprows=1, encoding='utf-8', sep=delimiter)
        
        # Supprimer les lignes complètement vides
        df = df.dropna(how='all')
        
        # Standardiser les colonnes
        df, mapping, std_errors = standardize_sf_columns(df)
        errors.extend(std_errors)
        
        if errors:
            return df, mapping, errors
        
        # Nettoyage des URLs
        if 'url' in df.columns:
            df['url'] = df['url'].apply(clean_url)
        
        # Conversion des colonnes numériques
        numeric_cols = ['status_code', 'title_length', 'meta_desc_length', 'word_count', 
                       'crawl_depth', 'inlinks', 'unique_inlinks', 'outlinks', 
                       'unique_outlinks', 'response_time', 'h1_length']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calcul de la profondeur si non présente
        if 'crawl_depth' not in df.columns or df['crawl_depth'].isna().all():
            df['crawl_depth'] = df['url'].apply(calculate_crawl_depth)
        
        return df, mapping, errors
    
    except Exception as e:
        errors.append(f"Erreur lors du chargement: {str(e)}")
        return pd.DataFrame(), {}, errors


@st.cache_data
def load_gsc_404_data(file) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Charge et prépare les données GSC des pages 404.
    Export depuis: GSC > Indexation > Pages > Introuvable (404)
    """
    errors = []
    
    try:
        # Lecture du fichier
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            content = file.read()
            file.seek(0)
            
            # Détecter le délimiteur
            first_lines = content.decode('utf-8', errors='ignore')[:2000]
            if '\t' in first_lines and first_lines.count('\t') > first_lines.count(','):
                delimiter = '\t'
            elif ';' in first_lines and first_lines.count(';') > first_lines.count(','):
                delimiter = ';'
            else:
                delimiter = ','
            
            try:
                df = pd.read_csv(file, encoding='utf-8', sep=delimiter)
            except:
                file.seek(0)
                df = pd.read_csv(file, encoding='latin-1', sep=delimiter)
        
        # Supprimer les lignes complètement vides
        df = df.dropna(how='all')
        
        # Standardiser les colonnes
        df, mapping, std_errors = standardize_gsc_indexation_columns(df, '404')
        errors.extend(std_errors)
        
        if errors:
            return df, mapping, errors
        
        # Nettoyage des URLs
        if 'url' in df.columns:
            df['url'] = df['url'].apply(clean_url)
        
        # Ajouter une colonne pour identifier la source
        df['gsc_404'] = True
        
        return df, mapping, errors
    
    except Exception as e:
        errors.append(f"Erreur lors du chargement du fichier 404: {str(e)}")
        return pd.DataFrame(), {}, errors


@st.cache_data
def load_gsc_redirects_data(file) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Charge et prépare les données GSC des pages avec redirections.
    Export depuis: GSC > Indexation > Pages > Page avec redirection
    """
    errors = []
    
    try:
        # Lecture du fichier
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            content = file.read()
            file.seek(0)
            
            # Détecter le délimiteur
            first_lines = content.decode('utf-8', errors='ignore')[:2000]
            if '\t' in first_lines and first_lines.count('\t') > first_lines.count(','):
                delimiter = '\t'
            elif ';' in first_lines and first_lines.count(';') > first_lines.count(','):
                delimiter = ';'
            else:
                delimiter = ','
            
            try:
                df = pd.read_csv(file, encoding='utf-8', sep=delimiter)
            except:
                file.seek(0)
                df = pd.read_csv(file, encoding='latin-1', sep=delimiter)
        
        # Supprimer les lignes complètement vides
        df = df.dropna(how='all')
        
        # Standardiser les colonnes
        df, mapping, std_errors = standardize_gsc_indexation_columns(df, 'redirections')
        errors.extend(std_errors)
        
        if errors:
            return df, mapping, errors
        
        # Nettoyage des URLs
        if 'url' in df.columns:
            df['url'] = df['url'].apply(clean_url)
        
        # Ajouter une colonne pour identifier la source
        df['gsc_redirect'] = True
        
        return df, mapping, errors
    
    except Exception as e:
        errors.append(f"Erreur lors du chargement du fichier redirections: {str(e)}")
        return pd.DataFrame(), {}, errors


# =============================================================================
# FONCTIONS D'ANALYSE INDEXATION GSC (404 & REDIRECTIONS)
# =============================================================================

def analyze_gsc_404(df_404: pd.DataFrame, df_sf: pd.DataFrame) -> Dict:
    """
    Analyse les pages 404 signalées par GSC et les croise avec Screaming Frog.
    
    Détecte:
    - 404 confirmées par le crawl SF
    - 404 maintenant corrigées (200 OK)
    - 404 maintenant redirigées
    - 404 avec liens internes pointant vers elles
    - 404 non présentes dans le crawl
    """
    results = {
        'total_404_gsc': len(df_404),
        'issues': [],
        'recommendations': [],
        'metrics': {},
        'details': {}
    }
    
    if df_404.empty:
        return results
    
    # Préparer les colonnes pour le croisement
    sf_cols = ['url', 'status_code']
    if 'unique_inlinks' in df_sf.columns:
        sf_cols.append('unique_inlinks')
    elif 'inlinks' in df_sf.columns:
        sf_cols.append('inlinks')
    
    # Croiser avec SF pour voir si ces 404 existent dans le crawl
    df_merged = pd.merge(
        df_404[['url']].drop_duplicates(),
        df_sf[sf_cols].drop_duplicates() if not df_sf.empty else pd.DataFrame(columns=sf_cols),
        on='url',
        how='left',
        suffixes=('_404', '_sf')
    )
    
    # Analyser les résultats du croisement
    if 'status_code' in df_merged.columns:
        # Pages 404 GSC présentes dans le crawl SF
        in_crawl = df_merged[df_merged['status_code'].notna()]
        not_in_crawl = df_merged[df_merged['status_code'].isna()]
        
        # Pages 404 GSC qui sont toujours 404 dans SF
        still_404 = in_crawl[in_crawl['status_code'] == 404]
        
        # Pages 404 GSC qui sont maintenant OK (200) dans SF - corrigées !
        now_200 = in_crawl[in_crawl['status_code'] == 200]
        
        # Pages 404 GSC maintenant redirigées
        now_redirected = in_crawl[in_crawl['status_code'].isin([301, 302, 307, 308])]
        
        # Pages 404 GSC avec des inlinks (liens internes pointant vers elles)
        inlinks_col = 'unique_inlinks' if 'unique_inlinks' in df_merged.columns else 'inlinks' if 'inlinks' in df_merged.columns else None
        with_inlinks = pd.DataFrame()
        if inlinks_col and inlinks_col in df_merged.columns:
            with_inlinks = df_merged[df_merged[inlinks_col].fillna(0) > 0]
        
        # Stocker les métriques
        results['metrics'] = {
            'total_404_gsc': len(df_404),
            'in_crawl': len(in_crawl),
            'not_in_crawl': len(not_in_crawl),
            'still_404': len(still_404),
            'now_200': len(now_200),
            'now_redirected': len(now_redirected),
            'with_inlinks': len(with_inlinks)
        }
        
        # Stocker les détails pour l'affichage
        results['details']['still_404'] = still_404.head(50).to_dict('records') if not still_404.empty else []
        results['details']['now_200'] = now_200.head(50).to_dict('records') if not now_200.empty else []
        results['details']['with_inlinks'] = with_inlinks.head(50).to_dict('records') if not with_inlinks.empty else []
        results['details']['not_in_crawl'] = not_in_crawl.head(50).to_dict('records') if not not_in_crawl.empty else []
        
        # Générer les issues
        if len(still_404) > 0:
            results['issues'].append({
                'severity': 'critical',
                'message': f"🔴 {len(still_404)} pages 404 GSC toujours en erreur (confirmées par crawl)",
                'impact': "Ces pages 404 sont actives et nuisent à l'indexation et l'expérience utilisateur"
            })
            results['recommendations'].append(
                "Mettre en place des redirections 301 vers des pages pertinentes ou recréer le contenu"
            )
        
        if len(with_inlinks) > 0:
            results['issues'].append({
                'severity': 'critical',
                'message': f"⚠️ {len(with_inlinks)} pages 404 GSC ont des liens internes pointant vers elles",
                'impact': "Ces liens cassés gaspillent le budget de crawl et créent une mauvaise UX"
            })
            results['recommendations'].append(
                "Corriger ou supprimer tous les liens internes pointant vers ces pages 404"
            )
        
        if len(now_200) > 0:
            results['issues'].append({
                'severity': 'info',
                'message': f"✅ {len(now_200)} pages 404 GSC sont maintenant en 200 OK (corrigées)",
                'impact': "Ces erreurs ont été corrigées, attendre la re-indexation par Google"
            })
        
        if len(now_redirected) > 0:
            results['issues'].append({
                'severity': 'info',
                'message': f"↪️ {len(now_redirected)} pages 404 GSC sont maintenant redirigées",
                'impact': "Ces redirections permettront à Google de mettre à jour son index"
            })
        
        if len(not_in_crawl) > 0:
            pct_not_crawled = round(len(not_in_crawl) / len(df_404) * 100, 1) if len(df_404) > 0 else 0
            results['issues'].append({
                'severity': 'medium',
                'message': f"❓ {len(not_in_crawl)} pages 404 GSC non trouvées dans le crawl SF ({pct_not_crawled}%)",
                'impact': "Ces URLs peuvent être des anciennes pages supprimées ou des URLs mal formées"
            })
    else:
        results['metrics'] = {
            'total_404_gsc': len(df_404),
            'in_crawl': 0,
            'not_in_crawl': len(df_404),
            'still_404': 0,
            'now_200': 0,
            'now_redirected': 0,
            'with_inlinks': 0
        }
    
    return results


def analyze_gsc_redirects(df_redirects: pd.DataFrame, df_sf: pd.DataFrame) -> Dict:
    """
    Analyse les redirections signalées par GSC et les croise avec Screaming Frog.
    
    Détecte:
    - Redirections 301 (permanentes) confirmées
    - Redirections 302 (temporaires) à convertir
    - Redirections menant à des 404 (chaînes cassées)
    - Redirections avec liens internes
    - Chaînes de redirections
    """
    results = {
        'total_redirects_gsc': len(df_redirects),
        'issues': [],
        'recommendations': [],
        'metrics': {},
        'details': {}
    }
    
    if df_redirects.empty:
        return results
    
    # Préparer les colonnes pour le croisement
    sf_cols = ['url', 'status_code']
    if 'redirect_url' in df_sf.columns:
        sf_cols.append('redirect_url')
    if 'unique_inlinks' in df_sf.columns:
        sf_cols.append('unique_inlinks')
    elif 'inlinks' in df_sf.columns:
        sf_cols.append('inlinks')
    
    # Croiser avec SF
    df_merged = pd.merge(
        df_redirects[['url']].drop_duplicates(),
        df_sf[sf_cols].drop_duplicates() if not df_sf.empty else pd.DataFrame(columns=sf_cols),
        on='url',
        how='left',
        suffixes=('_redir', '_sf')
    )
    
    # Analyser les résultats du croisement
    if 'status_code' in df_merged.columns:
        # Pages redirect GSC présentes dans le crawl SF
        in_crawl = df_merged[df_merged['status_code'].notna()]
        not_in_crawl = df_merged[df_merged['status_code'].isna()]
        
        # Différents types de redirections dans SF
        redirect_301 = in_crawl[in_crawl['status_code'].isin([301, 308])]
        redirect_302 = in_crawl[in_crawl['status_code'].isin([302, 307])]
        now_200 = in_crawl[in_crawl['status_code'] == 200]
        now_404 = in_crawl[in_crawl['status_code'] == 404]
        
        # Avec liens internes
        inlinks_col = 'unique_inlinks' if 'unique_inlinks' in df_merged.columns else 'inlinks' if 'inlinks' in df_merged.columns else None
        with_inlinks = pd.DataFrame()
        if inlinks_col and inlinks_col in df_merged.columns:
            with_inlinks = df_merged[df_merged[inlinks_col].fillna(0) > 0]
        
        # Stocker les métriques
        results['metrics'] = {
            'total_redirects_gsc': len(df_redirects),
            'in_crawl': len(in_crawl),
            'not_in_crawl': len(not_in_crawl),
            'redirect_301': len(redirect_301),
            'redirect_302': len(redirect_302),
            'now_200': len(now_200),
            'now_404': len(now_404),
            'with_inlinks': len(with_inlinks)
        }
        
        # Stocker les détails
        results['details']['redirect_301'] = redirect_301.head(50).to_dict('records') if not redirect_301.empty else []
        results['details']['redirect_302'] = redirect_302.head(50).to_dict('records') if not redirect_302.empty else []
        results['details']['now_404'] = now_404.head(50).to_dict('records') if not now_404.empty else []
        results['details']['with_inlinks'] = with_inlinks.head(50).to_dict('records') if not with_inlinks.empty else []
        
        # Générer les issues
        if len(redirect_302) > 0:
            results['issues'].append({
                'severity': 'high',
                'message': f"⚠️ {len(redirect_302)} redirections temporaires (302/307) détectées",
                'impact': "Les 302 ne transmettent pas tout le link juice - convertir en 301 si permanentes"
            })
            results['recommendations'].append(
                "Convertir les redirections 302/307 en 301 si elles sont permanentes"
            )
        
        if len(redirect_301) > 0:
            results['issues'].append({
                'severity': 'info',
                'message': f"✅ {len(redirect_301)} redirections permanentes (301/308) correctement configurées",
                'impact': "Configuration correcte, vérifier que les destinations sont pertinentes"
            })
        
        if len(now_404) > 0:
            results['issues'].append({
                'severity': 'critical',
                'message': f"🔴 {len(now_404)} redirections GSC mènent maintenant à des 404 !",
                'impact': "CRITIQUE: Chaînes de redirection cassées - très mauvais pour le SEO"
            })
            results['recommendations'].append(
                "URGENT: Corriger les chaînes de redirection cassées qui aboutissent à des 404"
            )
        
        if len(now_200) > 0:
            results['issues'].append({
                'severity': 'medium',
                'message': f"🔄 {len(now_200)} redirections GSC répondent maintenant en 200 OK",
                'impact': "Ces URLs ne redirigent plus - vérifier si c'est intentionnel"
            })
        
        if len(with_inlinks) > 0:
            results['issues'].append({
                'severity': 'medium',
                'message': f"🔗 {len(with_inlinks)} pages redirigées ont encore des liens internes",
                'impact': "Ces liens devraient pointer directement vers la destination finale"
            })
            results['recommendations'].append(
                "Mettre à jour les liens internes pour pointer vers les URLs finales"
            )
        
        # Analyse des chaînes de redirection si disponible
        if 'redirect_url' in df_sf.columns and not df_sf.empty:
            redirect_chains = df_sf[df_sf['status_code'].isin([301, 302, 307, 308])]
            if not redirect_chains.empty:
                destinations = redirect_chains['redirect_url'].apply(clean_url)
                redirect_sources = set(redirect_chains['url'].apply(clean_url))
                chain_redirects = destinations.isin(redirect_sources).sum()
                
                if chain_redirects > 0:
                    results['issues'].append({
                        'severity': 'high',
                        'message': f"🔗 {chain_redirects} chaînes de redirections détectées",
                        'impact': "Les chaînes ralentissent le crawl et diluent le link juice"
                    })
                    results['recommendations'].append(
                        "Simplifier les chaînes en redirigeant directement vers la destination finale"
                    )
                    results['metrics']['redirect_chains'] = chain_redirects
    else:
        results['metrics'] = {
            'total_redirects_gsc': len(df_redirects),
            'in_crawl': 0,
            'not_in_crawl': len(df_redirects),
            'redirect_301': 0,
            'redirect_302': 0,
            'now_200': 0,
            'now_404': 0,
            'with_inlinks': 0
        }
    
    return results

def analyze_status_codes(df: pd.DataFrame) -> Dict:
    """Analyse complète des codes de statut."""
    results = {
        'total': len(df),
        'distribution': {},
        'issues': [],
        'recommendations': [],
        'metrics': {}
    }
    
    if 'status_code' not in df.columns:
        return results
    
    status_counts = df['status_code'].value_counts().to_dict()
    results['distribution'] = status_counts
    
    total = len(df)
    code_200 = sum(v for k, v in status_counts.items() if pd.notna(k) and int(k) == 200)
    redirects_301 = sum(v for k, v in status_counts.items() if pd.notna(k) and int(k) in [301, 308])
    redirects_302 = sum(v for k, v in status_counts.items() if pd.notna(k) and int(k) in [302, 307])
    errors_4xx = sum(v for k, v in status_counts.items() if pd.notna(k) and 400 <= int(k) < 500)
    errors_5xx = sum(v for k, v in status_counts.items() if pd.notna(k) and 500 <= int(k) < 600)
    
    results['metrics'] = {
        '200_ok': code_200,
        '200_pct': round(code_200 / total * 100, 1) if total > 0 else 0,
        'redirects_301': redirects_301,
        'redirects_302': redirects_302,
        'errors_4xx': errors_4xx,
        'errors_5xx': errors_5xx
    }
    
    if errors_5xx > 0:
        results['issues'].append({
            'severity': 'critical',
            'message': f"🔥 {errors_5xx} erreurs serveur (5xx) détectées",
            'impact': "Les erreurs 5xx empêchent l'indexation et dégradent l'expérience utilisateur"
        })
    
    if errors_4xx > 0:
        pct_4xx = round(errors_4xx / total * 100, 1)
        severity = 'critical' if pct_4xx > 10 else 'high' if pct_4xx > 5 else 'medium'
        results['issues'].append({
            'severity': severity,
            'message': f"❌ {errors_4xx} erreurs client (4xx) - {pct_4xx}% du crawl",
            'impact': "Les erreurs 4xx créent des impasses pour les bots et les utilisateurs"
        })
    
    if redirects_302 > 0:
        results['issues'].append({
            'severity': 'medium',
            'message': f"↩️ {redirects_302} redirections temporaires (302/307)",
            'impact': "Les redirections temporaires ne transmettent pas tout le link juice"
        })
    
    if redirects_301 > total * 0.15:
        results['issues'].append({
            'severity': 'medium',
            'message': f"↪️ Taux élevé de redirections 301: {redirects_301} ({round(redirects_301/total*100, 1)}%)",
            'impact': "Trop de redirections diluent le budget de crawl"
        })
    
    return results


def analyze_indexability(df: pd.DataFrame) -> Dict:
    """Analyse de l'indexabilité des pages."""
    results = {
        'total': len(df),
        'indexable': 0,
        'non_indexable': 0,
        'reasons': {},
        'issues': [],
        'recommendations': []
    }
    
    if 'indexability' not in df.columns:
        if 'status_code' in df.columns:
            results['indexable'] = len(df[df['status_code'] == 200])
            results['non_indexable'] = len(df) - results['indexable']
        return results
    
    # Détecter les pages indexables
    indexable_mask = df['indexability'].astype(str).str.lower().str.strip() == 'indexable'
    
    results['indexable'] = indexable_mask.sum()
    results['non_indexable'] = len(df) - results['indexable']
    
    if 'indexability_status' in df.columns:
        non_indexable_df = df[~indexable_mask]
        reasons = non_indexable_df['indexability_status'].value_counts().to_dict()
        results['reasons'] = reasons
    
    indexability_ratio = results['indexable'] / results['total'] * 100 if results['total'] > 0 else 0
    if indexability_ratio < 70:
        results['issues'].append({
            'severity': 'high',
            'message': f"📊 Seulement {round(indexability_ratio, 1)}% des pages sont indexables",
            'impact': "Une grande partie du site n'est pas visible dans les moteurs de recherche"
        })
    
    return results


def analyze_titles(df: pd.DataFrame) -> Dict:
    """Analyse complète des balises title."""
    results = {
        'total': len(df),
        'metrics': {},
        'issues': [],
        'recommendations': [],
        'duplicates': {}
    }
    
    title_col = 'title' if 'title' in df.columns else None
    length_col = 'title_length' if 'title_length' in df.columns else None
    
    if not title_col and not length_col:
        return results
    
    if length_col:
        missing = len(df[df[length_col].fillna(0) == 0])
        too_short = len(df[(df[length_col] > 0) & (df[length_col] < 30)])
        optimal = len(df[(df[length_col] >= 30) & (df[length_col] <= 60)])
        too_long = len(df[df[length_col] > 70])
        
        results['metrics'] = {
            'missing': missing,
            'too_short': too_short,
            'optimal': optimal,
            'too_long': too_long
        }
        
        if missing > 0:
            results['issues'].append({
                'severity': 'critical',
                'message': f"❌ {missing} pages sans balise title",
                'impact': "Les pages sans title ont un désavantage majeur en SEO"
            })
        
        if too_long > len(df) * 0.2:
            results['issues'].append({
                'severity': 'medium',
                'message': f"✂️ {too_long} titles trop longs (>70 car.)",
                'impact': "Les titles trop longs sont tronqués dans les SERP"
            })
    
    if title_col:
        df_with_titles = df[df[title_col].notna() & (df[title_col] != '')]
        if len(df_with_titles) > 0:
            title_counts = df_with_titles[title_col].value_counts()
            duplicates = title_counts[title_counts > 1]
            if len(duplicates) > 0:
                dup_count = len(duplicates)
                affected_pages = duplicates.sum()
                results['metrics']['duplicates_count'] = dup_count
                
                # Stocker quelques exemples
                for title in duplicates.head(10).index:
                    urls = df_with_titles[df_with_titles[title_col] == title]['url'].tolist()
                    results['duplicates'][title] = urls[:5]
                
                results['issues'].append({
                    'severity': 'high',
                    'message': f"📋 {dup_count} titles en doublon affectant {affected_pages} pages",
                    'impact': "Les titles dupliqués créent de la confusion pour Google"
                })
    
    return results


def analyze_meta_descriptions(df: pd.DataFrame) -> Dict:
    """Analyse complète des meta descriptions."""
    results = {
        'total': len(df),
        'metrics': {},
        'issues': [],
        'recommendations': [],
        'duplicates': {}
    }
    
    desc_col = 'meta_description' if 'meta_description' in df.columns else None
    length_col = 'meta_desc_length' if 'meta_desc_length' in df.columns else None
    
    if not desc_col and not length_col:
        return results
    
    if length_col:
        missing = len(df[df[length_col].fillna(0) == 0])
        too_short = len(df[(df[length_col] > 0) & (df[length_col] < 70)])
        optimal = len(df[(df[length_col] >= 70) & (df[length_col] <= 155)])
        too_long = len(df[df[length_col] > 160])
        
        results['metrics'] = {
            'missing': missing,
            'too_short': too_short,
            'optimal': optimal,
            'too_long': too_long
        }
        
        if missing > len(df) * 0.1:
            results['issues'].append({
                'severity': 'medium',
                'message': f"📝 {missing} pages sans meta description ({round(missing/len(df)*100, 1)}%)",
                'impact': "Google peut générer des descriptions automatiques moins optimisées"
            })
    
    if desc_col:
        df_with_meta = df[df[desc_col].notna() & (df[desc_col] != '')]
        if len(df_with_meta) > 0:
            meta_counts = df_with_meta[desc_col].value_counts()
            duplicates = meta_counts[meta_counts > 1]
            if len(duplicates) > 5:
                results['metrics']['duplicates_count'] = len(duplicates)
                results['issues'].append({
                    'severity': 'medium',
                    'message': f"📋 {len(duplicates)} meta descriptions en doublon",
                    'impact': "Opportunité manquée de personnaliser l'extrait dans les SERP"
                })
    
    return results


def analyze_headings(df: pd.DataFrame) -> Dict:
    """Analyse des balises H1."""
    results = {
        'total': len(df),
        'metrics': {},
        'issues': [],
        'recommendations': []
    }
    
    if 'h1' not in df.columns:
        return results
    
    missing_h1 = len(df[df['h1'].isna() | (df['h1'] == '')])
    results['metrics']['missing_h1'] = missing_h1
    
    if missing_h1 > 0:
        pct = round(missing_h1 / len(df) * 100, 1)
        severity = 'critical' if pct > 20 else 'high' if pct > 10 else 'medium'
        results['issues'].append({
            'severity': severity,
            'message': f"🏷️ {missing_h1} pages sans H1 ({pct}%)",
            'impact': "Le H1 est crucial pour la compréhension du sujet de la page"
        })
    
    # Doublons H1
    df_with_h1 = df[df['h1'].notna() & (df['h1'] != '')]
    if len(df_with_h1) > 0:
        h1_counts = df_with_h1['h1'].value_counts()
        duplicates = h1_counts[h1_counts > 1]
        if len(duplicates) > 0:
            results['metrics']['h1_duplicates'] = len(duplicates)
            results['issues'].append({
                'severity': 'medium',
                'message': f"📋 {len(duplicates)} H1 en doublon",
                'impact': "Des H1 identiques indiquent possiblement du contenu dupliqué"
            })
    
    return results


def analyze_content(df: pd.DataFrame) -> Dict:
    """Analyse du contenu (word count)."""
    results = {
        'total': len(df),
        'metrics': {},
        'issues': [],
        'recommendations': []
    }
    
    if 'word_count' not in df.columns:
        return results
    
    empty_pages = len(df[df['word_count'].fillna(0) == 0])
    very_thin = len(df[(df['word_count'] > 0) & (df['word_count'] < 100)])
    thin = len(df[(df['word_count'] >= 100) & (df['word_count'] < 300)])
    optimal = len(df[(df['word_count'] >= 600) & (df['word_count'] <= 1500)])
    rich = len(df[df['word_count'] > 1500])
    
    avg_word_count = df['word_count'].mean()
    median_word_count = df['word_count'].median()
    
    results['metrics'] = {
        'empty': empty_pages,
        'very_thin': very_thin,
        'thin': thin,
        'optimal': optimal,
        'rich': rich,
        'average': round(avg_word_count, 0) if pd.notna(avg_word_count) else 0,
        'median': round(median_word_count, 0) if pd.notna(median_word_count) else 0
    }
    
    thin_total = empty_pages + very_thin + thin
    thin_pct = round(thin_total / len(df) * 100, 1) if len(df) > 0 else 0
    
    if thin_pct > 30:
        results['issues'].append({
            'severity': 'critical',
            'message': f"📄 {thin_total} pages thin content ({thin_pct}%)",
            'impact': "Le thin content est un signal négatif fort pour Google"
        })
    elif thin_pct > 15:
        results['issues'].append({
            'severity': 'high',
            'message': f"📄 {thin_total} pages avec peu de contenu ({thin_pct}%)",
            'impact': "Ces pages apportent peu de valeur aux utilisateurs"
        })
    
    if empty_pages > 0:
        results['issues'].append({
            'severity': 'critical',
            'message': f"🚫 {empty_pages} pages vides (0 mots)",
            'impact': "Pages inutiles qui gaspillent le budget de crawl"
        })
    
    return results


def analyze_internal_linking(df: pd.DataFrame) -> Dict:
    """Analyse du maillage interne."""
    results = {
        'total': len(df),
        'metrics': {},
        'issues': [],
        'recommendations': []
    }
    
    inlinks_col = None
    if 'unique_inlinks' in df.columns:
        inlinks_col = 'unique_inlinks'
    elif 'inlinks' in df.columns:
        inlinks_col = 'inlinks'
    
    if not inlinks_col:
        return results
    
    avg_inlinks = df[inlinks_col].mean()
    median_inlinks = df[inlinks_col].median()
    orphan_pages = len(df[df[inlinks_col].fillna(0) <= 1])
    well_linked = len(df[df[inlinks_col] >= 10])
    
    results['metrics'] = {
        'average_inlinks': round(avg_inlinks, 1) if pd.notna(avg_inlinks) else 0,
        'median_inlinks': round(median_inlinks, 1) if pd.notna(median_inlinks) else 0,
        'orphan_pages': orphan_pages,
        'well_linked': well_linked
    }
    
    if orphan_pages > 0:
        pct = round(orphan_pages / len(df) * 100, 1)
        severity = 'critical' if pct > 10 else 'high' if pct > 5 else 'medium'
        results['issues'].append({
            'severity': severity,
            'message': f"🏝️ {orphan_pages} pages orphelines ({pct}%)",
            'impact': "Les pages orphelines sont difficiles à découvrir pour les bots"
        })
    
    if avg_inlinks and avg_inlinks < 5:
        results['issues'].append({
            'severity': 'medium',
            'message': f"🔗 Maillage interne faible (moyenne: {round(avg_inlinks, 1)} liens/page)",
            'impact': "Un maillage faible limite la distribution du PageRank"
        })
    
    return results


def analyze_crawl_depth(df: pd.DataFrame) -> Dict:
    """Analyse de la profondeur de crawl."""
    results = {
        'total': len(df),
        'metrics': {},
        'issues': [],
        'recommendations': []
    }
    
    if 'crawl_depth' not in df.columns:
        return results
    
    avg_depth = df['crawl_depth'].mean()
    max_depth = df['crawl_depth'].max()
    very_deep = len(df[df['crawl_depth'] > 5])
    
    results['metrics'] = {
        'average': round(avg_depth, 1) if pd.notna(avg_depth) else 0,
        'max': int(max_depth) if pd.notna(max_depth) else 0,
        'very_deep': very_deep
    }
    
    if very_deep > 0:
        results['issues'].append({
            'severity': 'high',
            'message': f"📉 {very_deep} pages à profondeur >5 clics",
            'impact': "Les pages profondes sont moins crawlées et moins prioritaires"
        })
    
    if avg_depth and avg_depth > 3:
        results['issues'].append({
            'severity': 'medium',
            'message': f"🌲 Profondeur moyenne élevée: {round(avg_depth, 1)} clics",
            'impact': "Une architecture trop profonde dilue l'autorité"
        })
    
    return results


# =============================================================================
# ANALYSE CROISÉE GSC + SCREAMING FROG
# =============================================================================

def cross_analyze_data(df_sf: pd.DataFrame, df_gsc: pd.DataFrame, 
                       df_404: pd.DataFrame = None, df_redirects: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Croise les données SF et GSC pour une analyse approfondie.
    
    Args:
        df_sf: DataFrame Screaming Frog
        df_gsc: DataFrame GSC Performance
        df_404: DataFrame GSC Indexation 404 (optionnel)
        df_redirects: DataFrame GSC Indexation Redirections (optionnel)
    
    Returns:
        Tuple contenant le DataFrame fusionné et le dictionnaire des résultats
    """
    
    results = {
        'total_sf': len(df_sf),
        'total_gsc': len(df_gsc),
        'matched': 0,
        'orphans_seo': 0,
        'zombies': 0,
        'quick_wins': 0,
        'issues': [],
        'recommendations': [],
        'top_opportunities': [],
        # Nouvelles clés pour les données d'indexation
        'gsc_404_analysis': {},
        'gsc_redirects_analysis': {}
    }
    
    # Merge des données
    df_merged = pd.merge(df_sf, df_gsc, on='url', how='left', suffixes=('', '_gsc'))
    
    results['matched'] = len(df_merged[df_merged['clicks'].notna()]) if 'clicks' in df_merged.columns else 0
    
    # Identifier les pages indexables (200 OK)
    if 'status_code' in df_merged.columns:
        indexable_mask = df_merged['status_code'] == 200
    else:
        indexable_mask = pd.Series([True] * len(df_merged))
    
    # Pages zombies
    if 'impressions' in df_merged.columns and 'clicks' in df_merged.columns:
        zombies_mask = (
            indexable_mask & 
            (df_merged['impressions'] > 100) & 
            (df_merged['clicks'] == 0)
        )
        results['zombies'] = zombies_mask.sum()
        
        if results['zombies'] > 0:
            results['issues'].append({
                'severity': 'high',
                'message': f"🧟 {results['zombies']} pages zombies (impressions >100, 0 clics)",
                'impact': "Ces pages sont visibles mais ne génèrent aucun trafic"
            })
    
    # Pages orphelines SEO
    if 'impressions' in df_merged.columns:
        orphans_mask = (
            indexable_mask & 
            (df_merged['impressions'].isna() | (df_merged['impressions'] == 0))
        )
        results['orphans_seo'] = orphans_mask.sum()
        
        if results['orphans_seo'] > len(df_sf) * 0.3:
            results['issues'].append({
                'severity': 'critical',
                'message': f"👻 {results['orphans_seo']} pages indexables sans impression ({round(results['orphans_seo']/len(df_sf)*100, 1)}%)",
                'impact': "Ces pages ne sont probablement pas indexées"
            })
    
    # Quick wins
    if 'position' in df_merged.columns and 'impressions' in df_merged.columns:
        quick_wins_mask = (
            indexable_mask &
            (df_merged['position'] >= 5) & 
            (df_merged['position'] <= 20) &
            (df_merged['impressions'] > 500)
        )
        quick_wins = df_merged[quick_wins_mask].sort_values('impressions', ascending=False)
        results['quick_wins'] = len(quick_wins)
        
        if results['quick_wins'] > 0:
            results['issues'].append({
                'severity': 'info',
                'message': f"💎 {results['quick_wins']} opportunités quick wins (pos. 5-20, >500 imp.)",
                'impact': "Ces pages peuvent facilement gagner des positions"
            })
            
            cols_to_show = ['url', 'impressions', 'clicks', 'position']
            cols_available = [c for c in cols_to_show if c in quick_wins.columns]
            results['top_opportunities'] = quick_wins.head(10)[cols_available].to_dict('records')
    
    # =========================================================================
    # ANALYSE DES DONNÉES D'INDEXATION GSC (NOUVEAU)
    # =========================================================================
    
    # Analyse des 404 GSC
    if df_404 is not None and not df_404.empty:
        results['gsc_404_analysis'] = analyze_gsc_404(df_404, df_sf)
        # Ajouter les issues au résumé global
        results['issues'].extend(results['gsc_404_analysis'].get('issues', []))
        results['recommendations'].extend(results['gsc_404_analysis'].get('recommendations', []))
    
    # Analyse des Redirections GSC
    if df_redirects is not None and not df_redirects.empty:
        results['gsc_redirects_analysis'] = analyze_gsc_redirects(df_redirects, df_sf)
        # Ajouter les issues au résumé global
        results['issues'].extend(results['gsc_redirects_analysis'].get('issues', []))
        results['recommendations'].extend(results['gsc_redirects_analysis'].get('recommendations', []))
    
    return df_merged, results


# =============================================================================
# GÉNÉRATION DU RAPPORT IA
# =============================================================================

def generate_ai_analysis(client: anthropic.Anthropic, data_summary: Dict, detailed_issues: List) -> str:
    """Génère une analyse experte via Claude API, incluant les données d'indexation GSC."""
    
    # Construire la section sur les 404 GSC si disponible
    gsc_404_section = ""
    if data_summary.get('gsc_404_total', 0) > 0:
        gsc_404_section = f"""
### Pages 404 (Rapport GSC Indexation)
- Total pages 404 signalées par GSC: {data_summary.get('gsc_404_total', 0)}
- Toujours en 404 (confirmé par crawl): {data_summary.get('gsc_404_still_404', 'N/A')}
- Maintenant corrigées (200 OK): {data_summary.get('gsc_404_now_200', 'N/A')}
- Maintenant redirigées: {data_summary.get('gsc_404_now_redirected', 'N/A')}
- Avec liens internes pointant vers elles: {data_summary.get('gsc_404_with_inlinks', 'N/A')}
- Non trouvées dans le crawl: {data_summary.get('gsc_404_not_crawled', 'N/A')}
"""
    
    # Construire la section sur les redirections GSC si disponible
    gsc_redirects_section = ""
    if data_summary.get('gsc_redirects_total', 0) > 0:
        gsc_redirects_section = f"""
### Redirections (Rapport GSC Indexation)
- Total redirections signalées par GSC: {data_summary.get('gsc_redirects_total', 0)}
- Redirections 301 (permanentes): {data_summary.get('gsc_redirects_301', 'N/A')}
- Redirections 302 (temporaires): {data_summary.get('gsc_redirects_302', 'N/A')}
- Maintenant en 200 OK: {data_summary.get('gsc_redirects_now_200', 'N/A')}
- Devenues 404 (chaîne cassée): {data_summary.get('gsc_redirects_now_404', 'N/A')}
- Avec liens internes: {data_summary.get('gsc_redirects_with_inlinks', 'N/A')}
- Chaînes de redirections: {data_summary.get('gsc_redirects_chains', 'N/A')}
"""
    
    prompt = f"""Tu es un consultant SEO technique senior avec 15 ans d'expérience. 
Voici les données d'un audit technique complet incluant les rapports d'indexation de Google Search Console.

## DONNÉES DE L'AUDIT

### Vue d'ensemble
- Pages crawlées: {data_summary.get('total_sf', 'N/A')}
- Pages GSC (Performance): {data_summary.get('total_gsc', 'N/A')}
- Pages matchées: {data_summary.get('matched', 'N/A')}

### Codes HTTP (Crawl)
{data_summary.get('status_codes_summary', 'Non disponible')}

### Contenu
- Moyenne mots/page: {data_summary.get('avg_word_count', 'N/A')}
- Pages thin (<300 mots): {data_summary.get('thin_content', 'N/A')}

### Maillage
- Moyenne liens entrants: {data_summary.get('avg_inlinks', 'N/A')}
- Pages orphelines: {data_summary.get('orphan_pages', 'N/A')}

### Performance GSC
- Clics totaux: {data_summary.get('total_clicks', 'N/A')}
- Impressions: {data_summary.get('total_impressions', 'N/A')}
- CTR moyen: {data_summary.get('avg_ctr', 'N/A')}
- Position moyenne: {data_summary.get('avg_position', 'N/A')}

### Analyse croisée Performance
- Pages zombies: {data_summary.get('zombies', 'N/A')}
- Quick wins: {data_summary.get('quick_wins', 'N/A')}
{gsc_404_section}
{gsc_redirects_section}

### Tous les problèmes détectés
{chr(10).join([f"- [{issue['severity'].upper()}] {issue['message']}" for issue in detailed_issues[:30]])}

## TA MISSION

Rédige un rapport d'audit SEO professionnel et complet incluant l'analyse des problèmes d'indexation (404 et redirections).

### 1. SYNTHÈSE EXÉCUTIVE (4-5 phrases)
Résume l'état global du site en intégrant les problèmes d'indexation détectés par Google.

### 2. SCORE DE SANTÉ TECHNIQUE (0-100)
Justifie en tenant compte des erreurs 404 et des problèmes de redirections.

### 3. PROBLÈMES CRITIQUES
Pour chaque problème, incluant les 404 et redirections:
- Impact business
- Complexité de résolution
- Quick win ou long terme

### 4. ANALYSE SPÉCIFIQUE INDEXATION
Détaille les actions à mener sur:
- Les pages 404 signalées par GSC (priorité aux 404 avec liens internes)
- Les redirections problématiques (302 à convertir, chaînes cassées)
- Les chaînes de redirection à simplifier

### 5. OPPORTUNITÉS RAPIDES

### 6. RECOMMANDATIONS STRATÉGIQUES (3-6 mois)

### 7. PLAN D'ACTION PRIORISÉ
Tableau: Action | Priorité | Effort | Impact

Sois particulièrement attentif aux problèmes d'indexation car ils impactent directement la visibilité du site dans Google.
Sois direct, actionnable et base-toi sur les données fournies.
"""
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=5000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Erreur lors de l'analyse IA: {str(e)}"


# =============================================================================
# VISUALISATIONS
# =============================================================================

def create_status_code_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Crée un graphique des codes de statut."""
    if 'status_code' not in df.columns:
        return None
    
    status_counts = df['status_code'].value_counts().sort_index()
    
    colors = []
    for code in status_counts.index:
        try:
            code_int = int(code)
            if code_int == 200:
                colors.append('#10B981')
            elif 300 <= code_int < 400:
                colors.append('#F59E0B')
            elif 400 <= code_int < 500:
                colors.append('#EF4444')
            elif 500 <= code_int < 600:
                colors.append('#7C3AED')
            else:
                colors.append('#6B7280')
        except:
            colors.append('#6B7280')
    
    fig = go.Figure(data=[
        go.Bar(
            x=[str(int(x)) if pd.notna(x) else 'N/A' for x in status_counts.index],
            y=status_counts.values,
            marker_color=colors,
            text=status_counts.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Distribution des codes de statut HTTP",
        xaxis_title="Code de statut",
        yaxis_title="Nombre de pages",
        showlegend=False,
        height=400
    )
    
    return fig


def create_content_distribution_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Crée un graphique de distribution du contenu."""
    if 'word_count' not in df.columns:
        return None
    
    bins = [0, 100, 300, 600, 1500, 3000, float('inf')]
    labels = ['Vide/Très thin', 'Thin (100-300)', 'Léger (300-600)', 
              'Optimal (600-1500)', 'Riche (1500-3000)', 'Très riche (>3000)']
    colors = ['#EF4444', '#F97316', '#F59E0B', '#10B981', '#059669', '#047857']
    
    df_copy = df.copy()
    df_copy['content_bin'] = pd.cut(df_copy['word_count'].fillna(0), bins=bins, labels=labels, right=False)
    content_dist = df_copy['content_bin'].value_counts().reindex(labels).fillna(0)
    
    fig = go.Figure(data=[
        go.Pie(
            labels=content_dist.index,
            values=content_dist.values,
            marker_colors=colors,
            hole=0.4,
            textinfo='percent+label'
        )
    ])
    
    fig.update_layout(
        title="Distribution du volume de contenu",
        height=400
    )
    
    return fig


def create_crawl_depth_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Crée un graphique de profondeur de crawl."""
    if 'crawl_depth' not in df.columns:
        return None
    
    depth_counts = df['crawl_depth'].value_counts().sort_index()
    
    colors = ['#10B981' if d <= 2 else '#F59E0B' if d <= 4 else '#EF4444' 
              for d in depth_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Niveau {int(x)}" for x in depth_counts.index],
            y=depth_counts.values,
            marker_color=colors,
            text=depth_counts.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Distribution de la profondeur de crawl",
        xaxis_title="Profondeur",
        yaxis_title="Nombre de pages",
        height=400
    )
    
    return fig


def create_gsc_performance_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Crée un graphique de performance GSC."""
    if 'clicks' not in df.columns or 'impressions' not in df.columns:
        return None
    
    # Top 20 pages par clics
    top_pages = df.nlargest(20, 'clicks').copy()
    top_pages['url_short'] = top_pages['url'].apply(lambda x: str(x)[-50:] if len(str(x)) > 50 else str(x))
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_pages['clicks'],
            y=top_pages['url_short'],
            orientation='h',
            marker_color='#667eea'
        )
    ])
    
    fig.update_layout(
        title="Top 20 pages par clics",
        xaxis_title="Clics",
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_ctr_by_position_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Crée un graphique CTR par position."""
    if 'position' not in df.columns or 'ctr' not in df.columns:
        return None
    
    df_filtered = df[(df['position'] >= 1) & (df['position'] <= 20)].copy()
    if len(df_filtered) == 0:
        return None
    
    df_filtered['position_round'] = df_filtered['position'].round()
    ctr_by_pos = df_filtered.groupby('position_round')['ctr'].mean().reset_index()
    
    benchmark_ctr = [0.28, 0.15, 0.11, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.025,
                    0.02, 0.018, 0.016, 0.015, 0.014, 0.013, 0.012, 0.011, 0.01, 0.009]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=ctr_by_pos['position_round'],
        y=ctr_by_pos['ctr'],
        name='Votre CTR',
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(1, 21)),
        y=benchmark_ctr,
        name='Benchmark',
        line=dict(color='#EF4444', dash='dash')
    ))
    
    fig.update_layout(
        title="CTR par position vs Benchmark",
        xaxis_title="Position",
        yaxis_title="CTR",
        yaxis_tickformat='.1%',
        height=400
    )
    
    return fig


def create_indexation_summary_chart(gsc_404_metrics: Dict, gsc_redirects_metrics: Dict) -> Optional[go.Figure]:
    """Crée un graphique résumant les problèmes d'indexation GSC."""
    
    categories = []
    values = []
    colors = []
    
    # Données 404
    if gsc_404_metrics:
        if gsc_404_metrics.get('total_404_gsc', 0) > 0:
            categories.append('404 GSC Total')
            values.append(gsc_404_metrics['total_404_gsc'])
            colors.append('#EF4444')
        
        if gsc_404_metrics.get('still_404', 0) > 0:
            categories.append('404 Confirmées')
            values.append(gsc_404_metrics['still_404'])
            colors.append('#DC2626')
        
        if gsc_404_metrics.get('with_inlinks', 0) > 0:
            categories.append('404 avec liens internes')
            values.append(gsc_404_metrics['with_inlinks'])
            colors.append('#B91C1C')
        
        if gsc_404_metrics.get('now_200', 0) > 0:
            categories.append('404 Corrigées (200)')
            values.append(gsc_404_metrics['now_200'])
            colors.append('#10B981')
    
    # Données Redirections
    if gsc_redirects_metrics:
        if gsc_redirects_metrics.get('total_redirects_gsc', 0) > 0:
            categories.append('Redirections GSC')
            values.append(gsc_redirects_metrics['total_redirects_gsc'])
            colors.append('#F59E0B')
        
        if gsc_redirects_metrics.get('redirect_302', 0) > 0:
            categories.append('Redir. 302 (temp.)')
            values.append(gsc_redirects_metrics['redirect_302'])
            colors.append('#D97706')
        
        if gsc_redirects_metrics.get('now_404', 0) > 0:
            categories.append('Redir. → 404 ⚠️')
            values.append(gsc_redirects_metrics['now_404'])
            colors.append('#7C3AED')
    
    if not categories:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Résumé des problèmes d'indexation GSC",
        xaxis_title="",
        yaxis_title="Nombre de pages",
        height=400,
        showlegend=False
    )
    
    return fig


# =============================================================================
# INTERFACE PRINCIPALE
# =============================================================================

def main():
    st.markdown('<p class="main-header">🔍 Audit SEO Technique Expert</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse approfondie combinant Screaming Frog + Google Search Console avec IA</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    api_key = st.sidebar.text_input(
        "Clé API Claude (sk-ant-...)",
        value=st.secrets.get("ANTHROPIC_API_KEY", ""), 
        type="password",
        help="Nécessaire pour l'analyse IA"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Paramètres")
    
    min_impressions = st.sidebar.number_input("Seuil min. impressions (zombies)", min_value=0, value=100)
    thin_content_threshold = st.sidebar.number_input("Seuil thin content (mots)", min_value=0, value=300)
    
    # Upload - Fichiers obligatoires
    st.markdown('<p class="section-title">📁 Import des données (obligatoires)</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Export Google Search Console")
        st.caption("⚠️ **Important**: Exporter depuis l'onglet **PAGES** (pas Dates)")
        gsc_file = st.file_uploader("Fichier GSC", type=['csv', 'xlsx'], key="gsc", label_visibility="collapsed")
        if gsc_file:
            st.success(f"✓ {gsc_file.name}")
    
    with col2:
        st.markdown("#### Export Screaming Frog")
        st.caption("Internal > HTML (CSV ou Excel)")
        sf_file = st.file_uploader("Fichier SF", type=['csv', 'xlsx'], key="sf", label_visibility="collapsed")
        if sf_file:
            st.success(f"✓ {sf_file.name}")
    
    # Upload - Fichiers d'indexation (optionnels)
    st.markdown('<p class="section-title">📁 Import des données d\'indexation GSC (optionnel mais recommandé)</p>', unsafe_allow_html=True)
    st.caption("Ces fichiers permettent une analyse croisée avec les données d'indexation de Google")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 🔴 Pages 404 (GSC)")
        st.caption("Indexation > Pages > Introuvable (404) > Exporter")
        gsc_404_file = st.file_uploader("Fichier 404 GSC", type=['csv', 'xlsx'], key="gsc_404", label_visibility="collapsed")
        if gsc_404_file:
            st.success(f"✓ {gsc_404_file.name}")
    
    with col4:
        st.markdown("#### ↪️ Pages avec redirections (GSC)")
        st.caption("Indexation > Pages > Page avec redirection > Exporter")
        gsc_redirects_file = st.file_uploader("Fichier Redirections GSC", type=['csv', 'xlsx'], key="gsc_redirects", label_visibility="collapsed")
        if gsc_redirects_file:
            st.success(f"✓ {gsc_redirects_file.name}")
    
    # Traitement
    if gsc_file and sf_file:
        with st.spinner("Chargement des données..."):
            df_gsc, gsc_mapping, gsc_errors = load_gsc_data(gsc_file)
            df_sf, sf_mapping, sf_errors = load_sf_data(sf_file)
            
            # Charger les fichiers d'indexation optionnels
            df_404 = pd.DataFrame()
            df_redirects = pd.DataFrame()
            gsc_404_errors = []
            gsc_redirects_errors = []
            
            if gsc_404_file:
                df_404, gsc_404_mapping, gsc_404_errors = load_gsc_404_data(gsc_404_file)
            
            if gsc_redirects_file:
                df_redirects, gsc_redirects_mapping, gsc_redirects_errors = load_gsc_redirects_data(gsc_redirects_file)
        
        # Afficher les erreurs si présentes
        if gsc_errors:
            for error in gsc_errors:
                st.error(error)
        
        if sf_errors:
            for error in sf_errors:
                st.error(error)
        
        if gsc_404_errors:
            for error in gsc_404_errors:
                st.error(error)
        
        if gsc_redirects_errors:
            for error in gsc_redirects_errors:
                st.error(error)
        
        # Debug info
        with st.expander("🔧 Debug - Colonnes détectées"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**GSC Performance - Mapping:**", gsc_mapping)
                st.write("**GSC Performance - Colonnes:**", list(df_gsc.columns)[:10] if not df_gsc.empty else "Vide")
                if not df_404.empty:
                    st.write(f"**GSC 404:** {len(df_404)} URLs chargées")
                if not df_redirects.empty:
                    st.write(f"**GSC Redirections:** {len(df_redirects)} URLs chargées")
            with col2:
                st.write("**SF - Mapping:**", sf_mapping)
                st.write("**SF - Colonnes:**", list(df_sf.columns)[:15] if not df_sf.empty else "Vide")
        
        # Stopper si erreurs critiques sur fichiers obligatoires
        if gsc_errors or sf_errors:
            st.stop()
        
        if df_gsc.empty or df_sf.empty:
            st.error("Les fichiers sont vides ou n'ont pas pu être chargés.")
            st.stop()
        
        # Vérification URL
        if 'url' not in df_gsc.columns or 'url' not in df_sf.columns:
            st.error("La colonne URL n'a pas été trouvée dans un des fichiers.")
            st.stop()
        
        # Analyse croisée complète (avec fichiers d'indexation optionnels)
        df_merged, cross_results = cross_analyze_data(
            df_sf, df_gsc, 
            df_404 if not df_404.empty else None,
            df_redirects if not df_redirects.empty else None
        )
        
        # Récupérer les métriques d'indexation pour l'affichage
        gsc_404_metrics = cross_results.get('gsc_404_analysis', {}).get('metrics', {})
        gsc_redirects_metrics = cross_results.get('gsc_redirects_analysis', {}).get('metrics', {})
        
        # KPIs
        st.markdown('<p class="section-title">📊 Vue d\'ensemble</p>', unsafe_allow_html=True)
        
        kpi_cols = st.columns(7)
        kpi_cols[0].metric("Pages crawlées", format_number(len(df_sf)))
        kpi_cols[1].metric("Pages GSC", format_number(len(df_gsc)))
        
        if 'clicks' in df_gsc.columns:
            kpi_cols[2].metric("Clics totaux", format_number(df_gsc['clicks'].sum()))
        
        kpi_cols[3].metric("Pages zombies", format_number(cross_results.get('zombies', 0)))
        kpi_cols[4].metric("Quick wins", format_number(cross_results.get('quick_wins', 0)))
        kpi_cols[5].metric("404 GSC", format_number(gsc_404_metrics.get('total_404_gsc', 0)) if gsc_404_metrics else "N/A")
        kpi_cols[6].metric("Redirections GSC", format_number(gsc_redirects_metrics.get('total_redirects_gsc', 0)) if gsc_redirects_metrics else "N/A")
        
        # Onglets - avec les nouveaux onglets pour 404 et redirections
        st.markdown('<p class="section-title">🔬 Analyses détaillées</p>', unsafe_allow_html=True)
        
        tabs = st.tabs(["🌐 HTTP", "📝 Balises", "📄 Contenu", "🔗 Maillage", "📈 GSC", 
                       "🔴 404 GSC", "↪️ Redirections GSC", "🔀 Croisée", "🤖 IA"])
        
        all_issues = []
        
        # Tab HTTP
        with tabs[0]:
            status_results = analyze_status_codes(df_sf)
            all_issues.extend(status_results.get('issues', []))
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = create_status_code_chart(df_sf)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Métriques")
                metrics = status_results.get('metrics', {})
                st.metric("Pages 200 OK", f"{metrics.get('200_ok', 0)} ({metrics.get('200_pct', 0)}%)")
                st.metric("Redirections 301", metrics.get('redirects_301', 0))
                st.metric("Erreurs 4xx", metrics.get('errors_4xx', 0))
                st.metric("Erreurs 5xx", metrics.get('errors_5xx', 0))
            
            for issue in status_results.get('issues', []):
                box_class = 'critical-box' if issue['severity'] == 'critical' else 'warning-box'
                st.markdown(f'<div class="{box_class}"><strong>{issue["message"]}</strong><br><em>{issue["impact"]}</em></div>', unsafe_allow_html=True)
        
        # Tab Balises
        with tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏷️ Titles")
                title_results = analyze_titles(df_sf)
                all_issues.extend(title_results.get('issues', []))
                
                metrics = title_results.get('metrics', {})
                if metrics:
                    mcols = st.columns(4)
                    mcols[0].metric("Manquants", metrics.get('missing', 0))
                    mcols[1].metric("Trop courts", metrics.get('too_short', 0))
                    mcols[2].metric("Optimaux", metrics.get('optimal', 0))
                    mcols[3].metric("Trop longs", metrics.get('too_long', 0))
            
            with col2:
                st.markdown("### 📝 Meta Descriptions")
                meta_results = analyze_meta_descriptions(df_sf)
                all_issues.extend(meta_results.get('issues', []))
                
                metrics = meta_results.get('metrics', {})
                if metrics:
                    mcols = st.columns(4)
                    mcols[0].metric("Manquantes", metrics.get('missing', 0))
                    mcols[1].metric("Trop courtes", metrics.get('too_short', 0))
                    mcols[2].metric("Optimales", metrics.get('optimal', 0))
                    mcols[3].metric("Trop longues", metrics.get('too_long', 0))
            
            st.markdown("### 🎯 H1")
            h1_results = analyze_headings(df_sf)
            all_issues.extend(h1_results.get('issues', []))
            
            h1_metrics = h1_results.get('metrics', {})
            hcols = st.columns(3)
            hcols[0].metric("H1 manquants", h1_metrics.get('missing_h1', 0))
            hcols[1].metric("H1 en doublon", h1_metrics.get('h1_duplicates', 0))
        
        # Tab Contenu
        with tabs[2]:
            content_results = analyze_content(df_sf)
            all_issues.extend(content_results.get('issues', []))
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_content_distribution_chart(df_sf)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Statistiques")
                metrics = content_results.get('metrics', {})
                st.metric("Moy. mots/page", format_number(metrics.get('average', 0)))
                st.metric("Pages vides", metrics.get('empty', 0))
                st.metric("Thin (<300)", metrics.get('thin', 0) + metrics.get('very_thin', 0))
                st.metric("Riches (>1500)", metrics.get('rich', 0))
            
            for issue in content_results.get('issues', []):
                box_class = 'critical-box' if issue['severity'] == 'critical' else 'warning-box'
                st.markdown(f'<div class="{box_class}"><strong>{issue["message"]}</strong></div>', unsafe_allow_html=True)
        
        # Tab Maillage
        with tabs[3]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔗 Maillage interne")
                linking_results = analyze_internal_linking(df_sf)
                all_issues.extend(linking_results.get('issues', []))
                
                metrics = linking_results.get('metrics', {})
                st.metric("Moy. liens entrants", metrics.get('average_inlinks', 0))
                st.metric("Pages orphelines", metrics.get('orphan_pages', 0))
                st.metric("Pages bien liées (>10)", metrics.get('well_linked', 0))
            
            with col2:
                st.markdown("### 📐 Profondeur")
                depth_results = analyze_crawl_depth(df_sf)
                all_issues.extend(depth_results.get('issues', []))
                
                fig = create_crawl_depth_chart(df_sf)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab GSC
        with tabs[4]:
            if 'clicks' in df_gsc.columns:
                st.markdown("### 📈 Performance GSC")
                
                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Clics", format_number(df_gsc['clicks'].sum()))
                kpi_cols[1].metric("Impressions", format_number(df_gsc['impressions'].sum()))
                if 'ctr' in df_gsc.columns:
                    kpi_cols[2].metric("CTR moyen", f"{df_gsc['ctr'].mean()*100:.2f}%")
                if 'position' in df_gsc.columns:
                    kpi_cols[3].metric("Position moyenne", f"{df_gsc['position'].mean():.1f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = create_gsc_performance_chart(df_merged)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = create_ctr_by_position_chart(df_gsc)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données GSC non disponibles.")
        
        # Tab Croisée
        with tabs[5]:
            st.markdown("### 🔀 Analyse croisée")
            all_issues.extend(cross_results.get('issues', []))
            
            mcols = st.columns(4)
            mcols[0].metric("Matchées", format_number(cross_results.get('matched', 0)))
            mcols[1].metric("Zombies", format_number(cross_results.get('zombies', 0)))
            mcols[2].metric("Orphelines SEO", format_number(cross_results.get('orphans_seo', 0)))
            mcols[3].metric("Quick wins", format_number(cross_results.get('quick_wins', 0)))
            
            for issue in cross_results.get('issues', []):
                box_class = 'success-box' if issue['severity'] == 'info' else 'critical-box' if issue['severity'] == 'critical' else 'warning-box'
                st.markdown(f'<div class="{box_class}"><strong>{issue["message"]}</strong><br><em>{issue["impact"]}</em></div>', unsafe_allow_html=True)
            
            if cross_results.get('top_opportunities'):
                st.markdown("#### 💎 Top Quick Wins")
                st.dataframe(pd.DataFrame(cross_results['top_opportunities']), use_container_width=True)
            
            # Export
            csv = df_merged.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger données croisées (CSV)", csv, 
                             f"audit_merged_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        
        # Tab IA
        with tabs[6]:
            st.markdown("### 🤖 Analyse IA")
            
            if not api_key:
                st.warning("⚠️ Entrez votre clé API Claude pour activer l'analyse IA.")
            else:
                data_summary = {
                    'total_sf': len(df_sf),
                    'total_gsc': len(df_gsc),
                    'matched': cross_results.get('matched', 0),
                    'zombies': cross_results.get('zombies', 0),
                    'quick_wins': cross_results.get('quick_wins', 0),
                }
                
                if 'status_code' in df_sf.columns:
                    status_dist = df_sf['status_code'].value_counts().to_dict()
                    data_summary['status_codes_summary'] = ', '.join([f"{int(k)}: {v}" for k, v in sorted(status_dist.items())])
                
                if 'word_count' in df_sf.columns:
                    data_summary['avg_word_count'] = round(df_sf['word_count'].mean(), 0)
                    data_summary['thin_content'] = len(df_sf[df_sf['word_count'] < thin_content_threshold])
                
                inlinks_col = 'unique_inlinks' if 'unique_inlinks' in df_sf.columns else 'inlinks' if 'inlinks' in df_sf.columns else None
                if inlinks_col:
                    data_summary['avg_inlinks'] = round(df_sf[inlinks_col].mean(), 1)
                    data_summary['orphan_pages'] = len(df_sf[df_sf[inlinks_col].fillna(0) <= 1])
                
                if 'clicks' in df_gsc.columns:
                    data_summary['total_clicks'] = int(df_gsc['clicks'].sum())
                    data_summary['total_impressions'] = int(df_gsc['impressions'].sum())
                    if 'ctr' in df_gsc.columns:
                        data_summary['avg_ctr'] = f"{df_gsc['ctr'].mean()*100:.2f}%"
                    if 'position' in df_gsc.columns:
                        data_summary['avg_position'] = round(df_gsc['position'].mean(), 1)
                
                if st.button("🚀 Générer le rapport IA", type="primary"):
                    with st.spinner("Claude analyse vos données..."):
                        try:
                            client = anthropic.Anthropic(api_key=api_key)
                            report = generate_ai_analysis(client, data_summary, all_issues)
                            st.markdown("---")
                            st.markdown(report)
                            
                            st.download_button("📄 Télécharger rapport (MD)", 
                                             report.encode('utf-8'),
                                             f"rapport_seo_{datetime.now().strftime('%Y%m%d')}.md",
                                             "text/markdown")
                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")
        
        # Résumé des issues
        st.markdown("---")
        st.markdown('<p class="section-title">📋 Tous les problèmes détectés</p>', unsafe_allow_html=True)
        
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        sorted_issues = sorted(all_issues, key=lambda x: severity_order.get(x['severity'], 5))
        
        if sorted_issues:
            for issue in sorted_issues:
                emoji = "🔴" if issue['severity'] == 'critical' else "🟠" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🟢" if issue['severity'] == 'low' else "🔵"
                st.markdown(f"{emoji} **[{issue['severity'].upper()}]** {issue['message']}")
        else:
            st.success("✅ Aucun problème majeur détecté !")
    
    else:
        st.info("👈 Importez vos fichiers pour commencer l'audit.")
        
        st.markdown("---")
        st.markdown("### 📖 Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📊 Export GSC
            1. Google Search Console > Performances
            2. **Onglet PAGES** (⚠️ pas Dates !)
            3. Exporter en CSV/Excel
            """)
        
        with col2:
            st.markdown("""
            #### 🐸 Export Screaming Frog
            1. Crawl du site
            2. Internal > HTML
            3. Export current tab
            """)
    
    st.markdown("---")
    st.caption("🔍 SEO Technical Audit Tool | Powered by Claude AI")


main()