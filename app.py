import streamlit as st
import torch

from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianTokenizer,
    MarianMTModel
)

from indictrans_telugu import IndicTransTelugu


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .caption-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #cccccc;
        font-size: 22px;
        font-weight: 500;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🖼️ AI Image Caption Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Generate AI-powered captions from images in English, Hindi and Telugu.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD BLIP MODEL
# =========================================================

@st.cache_resource
def load_caption_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model


# =========================================================
# LOAD HINDI MODEL
# =========================================================

@st.cache_resource
def load_hindi_model():

    model_name = "Helsinki-NLP/opus-mt-en-hi"

    tokenizer = MarianTokenizer.from_pretrained(
        model_name
    )

    model = MarianMTModel.from_pretrained(
        model_name
    )

    return tokenizer, model


# =========================================================
# LOAD TELUGU MODEL
# =========================================================

@st.cache_resource
def load_telugu_model():

    translator = IndicTransTelugu()

    return translator


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Settings")

language = st.sidebar.selectbox(
    "Choose Caption Language",
    [
        "English",
        "Hindi",
        "Telugu"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("### 🌐 Supported Languages")

st.sidebar.write("🇬🇧 English")
st.sidebar.write("🇮🇳 Hindi")
st.sidebar.write("🇮🇳 Telugu")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Upload an image and generate an AI-powered
    description in your selected language.
    """
)


# =========================================================
# IMAGE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# CAPTION GENERATION
# =========================================================

def generate_caption(image):

    processor, model = load_caption_model()

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption


# =========================================================
# HINDI TRANSLATION
# =========================================================

def translate_to_hindi(text):

    tokenizer, model = load_hindi_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        translated = model.generate(
            **inputs,
            max_length=100
        )

    result = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return result


# =========================================================
# CLEAN CAPTION
# =========================================================

def clean_caption(text):

    text = text.strip()

    # Remove unnecessary spaces before punctuation
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")

    return text


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # -------------------------------------------------
        # DISPLAY IMAGE
        # -------------------------------------------------

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            caption="Your Image",
            use_container_width=True
        )


        # -------------------------------------------------
        # GENERATE BUTTON
        # -------------------------------------------------

        if st.button(
            "✨ Generate Caption",
            use_container_width=True
        ):

            # =============================================
            # STEP 1 — BLIP CAPTION
            # =============================================

            with st.spinner(
                "🤖 AI is analyzing the image..."
            ):

                english_caption = generate_caption(
                    image
                )

            english_caption = clean_caption(
                english_caption
            )


            # =============================================
            # STEP 2 — SELECT LANGUAGE
            # =============================================

            # ---------------------------------------------
            # ENGLISH
            # ---------------------------------------------

            if language == "English":

                final_caption = english_caption

                language_label = "English"


            # ---------------------------------------------
            # HINDI
            # ---------------------------------------------

            elif language == "Hindi":

                with st.spinner(
                    "🌐 Translating caption to Hindi..."
                ):

                    final_caption = translate_to_hindi(
                        english_caption
                    )

                final_caption = clean_caption(
                    final_caption
                )

                language_label = "Hindi"


            # ---------------------------------------------
            # TELUGU
            # ---------------------------------------------

            elif language == "Telugu":

                with st.spinner(
                    "🌐 Translating caption to Telugu..."
                ):

                    telugu_translator = (
                        load_telugu_model()
                    )

                    final_caption = (
                        telugu_translator.translate(
                            english_caption
                        )
                    )

                final_caption = clean_caption(
                    final_caption
                )

                language_label = "Telugu"


            # =============================================
            # STEP 3 — DISPLAY CAPTION
            # =============================================

            st.subheader(
                f"🤖 AI Generated Caption ({language_label})"
            )

            st.markdown(
                f"""
                <div class="caption-box">
                    {final_caption}
                </div>
                """,
                unsafe_allow_html=True
            )


            # =============================================
            # STEP 4 — DOWNLOAD
            # =============================================

            st.download_button(
                label="📥 Download Caption",
                data=final_caption,
                file_name="image_caption.txt",
                mime="text/plain",
                use_container_width=True
            )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            "❌ Something went wrong while generating the caption."
        )

        st.exception(e)


# =========================================================
# ABOUT PROJECT
# =========================================================

with st.expander("ℹ️ About the Project"):

    st.write(
        """
        ### AI Image Caption Generator

        This project uses Artificial Intelligence to
        automatically generate a description of an image.

        The application uses a pre-trained BLIP
        (Bootstrapping Language-Image Pre-training) model
        for image caption generation.

        The generated English caption can be translated
        into Hindi or Telugu.
        """
    )


# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander("⚙️ How It Works"):

    st.write(
        """
        ### Step 1 — Upload Image

        The user uploads a JPG, JPEG or PNG image.

        ### Step 2 — Image Understanding

        The BLIP vision-language model analyzes the
        contents of the image.

        ### Step 3 — Caption Generation

        BLIP generates a natural-language description
        of the image in English.

        ### Step 4 — Translation

        If Hindi is selected, the caption is translated
        using an English-to-Hindi translation model.

        If Telugu is selected, the caption is translated
        using the IndicTrans2 ONNX model.

        ### Step 5 — Display

        The final caption is displayed in the selected
        language.

        ### Step 6 — Download

        The user can download the generated caption
        as a text file.
        """
    )


# =========================================================
# PROJECT TECHNOLOGIES
# =========================================================

with st.expander("🛠️ Technologies Used"):

    st.write(
        """
        **Programming Language**
        - Python

        **Computer Vision / Vision-Language Model**
        - BLIP

        **Natural Language Processing**
        - MarianMT
        - IndicTrans2

        **Deep Learning**
        - PyTorch
        - Transformers

        **Translation**
        - English → Hindi
        - English → Telugu

        **Interface**
        - Streamlit

        **Image Processing**
        - Pillow
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <small>
        AI Image Caption Generator | B.Tech AI Capstone Project
        </small>
    </div>
    """,
    unsafe_allow_html=True
)