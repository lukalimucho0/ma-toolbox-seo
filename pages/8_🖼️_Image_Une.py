import streamlit as st
from utils.auth import check_password
import anthropic
import google.generativeai as genai
from PIL import Image
import io
import re

st.set_page_config(
    page_title="Image à la Une",
    page_icon="🖼️",
    layout="wide"
)

check_password()


def inject_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0; }
    .prompt-box {
        background: #f8f9fa;
        border-left: 4px solid #7c3aed;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-family: monospace;
        font-size: 0.85rem;
        line-height: 1.5;
        color: #374151;
    }
    </style>
    """, unsafe_allow_html=True)


inject_css()

st.markdown("""
<div class="main-header">
    <h1>🖼️ Image à la Une</h1>
    <p>Prompt optimisé via Claude → image générée par Imagen 3 (Google AI)</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    anthropic_api_key = st.text_input(
        "Clé API Anthropic",
        value=st.secrets.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Pour générer le prompt Imagen optimisé via Claude"
    )

    gemini_api_key = st.text_input(
        "Clé API Gemini",
        value=st.secrets.get("GEMINI_API_KEY", ""),
        type="password",
        help="Google AI Studio → aistudio.google.com/app/apikey — pour Imagen 3"
    )

    st.divider()
    st.subheader("🎨 Paramètres image")

    STYLES = {
        "Photographie professionnelle": "professional photography, sharp focus, high quality DSLR, natural lighting",
        "Photo éditoriale magazine": "editorial magazine photography, journalistic style, cinematic lighting",
        "Illustration flat design": "flat design illustration, vector art, clean geometric shapes, bold colors",
        "Illustration minimaliste": "minimalist illustration, clean lines, simple shapes, pastel palette",
        "Design graphique moderne": "modern graphic design, bold composition, contemporary, no text",
        "Rendu 3D réaliste": "3D render, photorealistic, studio lighting, ray tracing, high detail",
    }

    style_label = st.selectbox("Style visuel", list(STYLES.keys()))

    aspect_ratio = st.selectbox(
        "Format",
        ["16:9", "4:3", "1:1", "3:4"],
        help="16:9 recommandé pour les images à la une de blog"
    )

    num_variants = st.slider("Nombre de variantes", 1, 4, 2)

    prompt_lang = st.selectbox(
        "Langue du prompt image",
        ["en", "fr"],
        help="'en' recommandé — meilleurs résultats avec Imagen 3"
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.subheader("📝 Sujet de l'article")

col_input, col_btn = st.columns([4, 1])
with col_input:
    h1 = st.text_input(
        "H1",
        placeholder="Ex : Comment réduire sa facture d'électricité en 2025",
        label_visibility="collapsed"
    )
with col_btn:
    run = st.button("🎨 Générer", type="primary", use_container_width=True, disabled=not h1)

extra_hints = st.text_input(
    "Précisions (optionnel)",
    placeholder="Ex : ambiance chaleureuse, couleurs bleues, sans personnages, extérieur ensoleillé…",
)

# ── Generation ────────────────────────────────────────────────────────────────
if run and h1:
    if not anthropic_api_key:
        st.error("❌ Clé API Anthropic manquante dans la sidebar ou les secrets Streamlit.")
        st.stop()
    if not gemini_api_key:
        st.error("❌ Clé API Gemini manquante dans la sidebar ou les secrets Streamlit.")
        st.stop()

    style_desc = STYLES[style_label]
    lang_word = "English" if prompt_lang == "en" else "French"

    # ── Step 1 : Claude → optimized Imagen prompts ────────────────────────────
    with st.status("🧠 Génération des prompts via Claude…", expanded=True) as status:
        try:
            claude = anthropic.Anthropic(api_key=anthropic_api_key)

            user_msg = f'Article title: "{h1}"'
            if extra_hints:
                user_msg += f"\nExtra guidance: {extra_hints}"

            msg = claude.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=900,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"You are an expert prompt engineer for Imagen 3 (Google AI image generation). "
                            f"Write {num_variants} distinct image prompt(s) in {lang_word} to illustrate a blog article. "
                            f"Style: {style_desc}. "
                            "Rules: NO text/words/letters/numbers in the image. "
                            "Describe scene, lighting, mood, colors, depth of field. Max 120 words per prompt. "
                            "Never show clearly identifiable real people or faces. "
                            "Output ONLY the prompts, one per line, prefixed by number (1. 2. etc.)\n\n"
                            + user_msg
                        ),
                    }
                ],
            )

            raw = msg.content[0].text.strip()
            prompts = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                if cleaned:
                    prompts.append(cleaned)
            prompts = prompts[:num_variants]

            status.update(label=f"✅ {len(prompts)} prompt(s) prêt(s)", state="running")

        except Exception as e:
            status.update(label=f"❌ Erreur Claude : {e}", state="error")
            st.stop()

        # ── Step 2 : Imagen 3 image generation ────────────────────────────────
        status.update(label="🎨 Génération des images via Imagen 3…", state="running")

        try:
            genai.configure(api_key=gemini_api_key)
            imagen = genai.ImageGenerationModel("imagen-3.0-generate-002")

            results = []
            for i, prompt in enumerate(prompts):
                status.update(label=f"🎨 Image {i + 1}/{len(prompts)} en cours…")
                img_response = imagen.generate_images(
                    prompt=prompt,
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level="block_some",
                    person_generation="allow_adult",
                )
                if img_response.images:
                    img_obj = img_response.images[0]
                    try:
                        pil_img = img_obj._pil_image
                    except Exception:
                        pil_img = Image.open(io.BytesIO(img_obj.image.image_bytes))
                    results.append((prompt, pil_img))

            status.update(label=f"✅ {len(results)} image(s) générée(s) !", state="complete")

        except Exception as e:
            status.update(label=f"❌ Erreur Imagen 3 : {e}", state="error")
            st.stop()

    # ── Display results ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("🖼️ Résultats")

    if not results:
        st.warning("Aucune image générée. Vérifie ta clé API Gemini et les filtres de sécurité.")
        st.stop()

    cols = st.columns(len(results))
    safe_name = re.sub(r"[^a-z0-9]+", "-", h1.lower())[:50].strip("-")

    for idx, (prompt, pil_img) in enumerate(results):
        with cols[idx]:
            st.image(pil_img, use_column_width=True, caption=f"Variante {idx + 1}")

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)

            st.download_button(
                label=f"⬇️ Télécharger v{idx + 1}",
                data=buf,
                file_name=f"image-une-{safe_name}-v{idx + 1}.png",
                mime="image/png",
                use_container_width=True,
                key=f"dl_{idx}"
            )

            with st.expander("📋 Prompt utilisé"):
                st.markdown(
                    f'<div class="prompt-box">{prompt}</div>',
                    unsafe_allow_html=True
                )
