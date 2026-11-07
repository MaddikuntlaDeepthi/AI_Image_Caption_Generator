
import streamlit as st
import torch
import hashlib

from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianTokenizer,
    MarianMTModel
)

from indictrans_telugu import IndicTransTelugu


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8f9fc;
}

.title {
    text-align: center;
    color: #1e3a5f;
    font-size: 36px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555555;
    font-size: 17px;
}

.caption-box {
    background-color: #eef4ff;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #2563eb;
    color: #172554;
    font-size: 18px;
    line-height: 1.6;
    margin-top: 15px;
}

.info-box {
    background-color: #f0f7ff;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# APPLICATION TITLE
# --------------------------------------------------

st.markdown(
    '<div class="title">🖼️ AI Image Caption Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Generate image captions in English, Hindi and Telugu using AI</div>',
    unsafe_allow_html=True
)

st.write("")


# --------------------------------------------------
# MODEL LOADING FUNCTIONS
# --------------------------------------------------

@st.cache_resource
def load_caption_model():

    model_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(model_name)

    model = BlipForConditionalGeneration.from_pretrained(
        model_name
    )

    model.eval()

    return processor, model


@st.cache_resource
def load_hindi_model():

    model_name = "Helsinki-NLP/opus-mt-en-hi"

    tokenizer = MarianTokenizer.from_pretrained(model_name)

    model = MarianMTModel.from_pretrained(model_name)

    model.eval()

    return tokenizer, model


@st.cache_resource
def load_telugu_model():

    translator = IndicTransTelugu()

    return translator


# --------------------------------------------------
# IMAGE CAPTION GENERATION
# --------------------------------------------------

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


# --------------------------------------------------
# HINDI TRANSLATION
# --------------------------------------------------

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


# --------------------------------------------------
# CAPTION CLEANING
# --------------------------------------------------

def clean_caption(text):

    text = text.strip()

    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")

    return text


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

if "english_caption" not in st.session_state:
    st.session_state.english_caption = None

if "final_caption" not in st.session_state:
    st.session_state.final_caption = None

if "caption_language" not in st.session_state:
    st.session_state.caption_language = None

if "image_hash" not in st.session_state:
    st.session_state.image_hash = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Settings")

language = st.sidebar.selectbox(
    "Choose Caption Language",
    ["English", "Hindi", "Telugu"]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload an image and generate its caption. "
    "You can switch languages without generating the caption again."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if uploaded_file is not None:

    try:

        image_bytes = uploaded_file.getvalue()

        current_image_hash = hashlib.sha256(
            image_bytes
        ).hexdigest()

        # Reset captions when a different image is uploaded
        if current_image_hash != st.session_state.image_hash:

            st.session_state.image_hash = current_image_hash

            st.session_state.english_caption = None

            st.session_state.final_caption = None

            st.session_state.caption_language = None

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            caption="Your Image",
            width="stretch"
        )

        st.write("")

        # --------------------------------------------------
        # GENERATE CAPTION BUTTON
        # --------------------------------------------------

        if st.button(
            "✨ Generate Caption",
            width="stretch"
        ):

            with st.spinner(
                "🤖 AI is analyzing the image..."
            ):

                english_caption = generate_caption(image)

                english_caption = clean_caption(
                    english_caption
                )

                st.session_state.english_caption = english_caption

                # Force translation into the currently selected language
                st.session_state.caption_language = None

        # --------------------------------------------------
        # AUTOMATIC LANGUAGE TRANSLATION
        # --------------------------------------------------

        if st.session_state.english_caption is not None:

            if language != st.session_state.caption_language:

                if language == "English":

                    st.session_state.final_caption = (
                        st.session_state.english_caption
                    )

                elif language == "Hindi":

                    with st.spinner(
                        "🌐 Translating caption to Hindi..."
                    ):

                        st.session_state.final_caption = (
                            translate_to_hindi(
                                st.session_state.english_caption
                            )
                        )

                elif language == "Telugu":

                    with st.spinner(
                        "🌐 Translating caption to Telugu..."
                    ):

                        telugu_translator = load_telugu_model()

                        st.session_state.final_caption = (
                            telugu_translator.translate(
                                st.session_state.english_caption
                            )
                        )

                st.session_state.final_caption = clean_caption(
                    st.session_state.final_caption
                )

                st.session_state.caption_language = language

            # --------------------------------------------------
            # DISPLAY FINAL CAPTION
            # --------------------------------------------------

            if st.session_state.final_caption:

                st.subheader(
                    f"🤖 AI Generated Caption ({language})"
                )

                st.markdown(
                    f"""
                    <div class="caption-box">
                        {st.session_state.final_caption}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                # --------------------------------------------------
                # DOWNLOAD CAPTION
                # --------------------------------------------------

                st.download_button(
                    label="📥 Download Caption",
                    data=st.session_state.final_caption,
                    file_name=f"image_caption_{language.lower()}.txt",
                    mime="text/plain",
                    width="stretch"
                )

    except Exception as e:

        st.error(
            "❌ Something went wrong while generating the caption."
        )

        st.exception(e)


else:

    st.info(
        "👆 Please upload an image to get started."
    )


# --------------------------------------------------
# ABOUT SECTION
# --------------------------------------------------

st.markdown("---")

with st.expander("ℹ️ About This Project"):

    st.write("""
    AI Image Caption Generator is an application that uses
    deep learning and natural language processing to generate
    meaningful descriptions of uploaded images.

    It supports three languages:

    - English
    - Hindi
    - Telugu
    """)


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

with st.expander("⚙️ How It Works"):

    st.write("""
    1. Upload an image.
    2. BLIP analyzes the image.
    3. An English caption is generated.
    4. MarianMT translates the caption into Hindi.
    5. IndicTrans2 translates the caption into Telugu.
    6. The final caption can be downloaded.
    """)


# --------------------------------------------------
# TECHNOLOGIES USED
# --------------------------------------------------

with st.expander("💻 Technologies Used"):

    st.write("""
    - Python
    - Streamlit
    - PyTorch
    - Hugging Face Transformers
    - BLIP Image Captioning Model
    - MarianMT Translation Model
    - IndicTrans2 ONNX
    - Pillow
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Developed using Artificial Intelligence and Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)
