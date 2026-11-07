# 🖼️ AI Image Caption Generator

An AI-powered multilingual application that automatically generates natural-language captions for uploaded images.

The application uses **BLIP (Bootstrapping Language-Image Pre-training)** for image caption generation and supports three languages:

- 🇬🇧 English
- 🇮🇳 Hindi
- 🇮🇳 Telugu

Built using Python, PyTorch, Hugging Face Transformers, IndicTrans2 ONNX, and Streamlit.

---

## 📌 Project Overview

Image captioning is the process of automatically generating textual descriptions of images using Artificial Intelligence.

This project combines:
- Computer Vision
- Deep Learning
- Natural Language Processing
- Vision-Language Models
- Machine Translation

The system analyzes an uploaded image, generates an English caption using BLIP, and optionally translates it into Hindi or Telugu.

## 🎯 Objectives

1. Automatically understand the content of an image.
2. Generate meaningful text descriptions.
3. Support multiple Indian languages.
4. Provide a simple, user-friendly interface.
5. Demonstrate practical applications of AI and Deep Learning.

## ✨ Features

- Upload images through a Streamlit interface.
- Generate captions automatically using BLIP.
- Display captions in English, Hindi, and Telugu.
- Translate captions using pretrained translation models.
- Download generated captions as `.txt` files.
- Simple and interactive web interface.

## 🏗️ System Architecture

```text
                       USER
                         |
                         v
                    Upload Image
                         |
                         v
                 BLIP Vision-Language
                       Model
                         |
                         v
                  English Caption
                         |
              +----------+----------+
              |          |          |
              v          v          v
           English    Hindi      Telugu
                        |           |
                        v           v
                     MarianMT   IndicTrans2
                                  ONNX
              |          |          |
              +----------+----------+
                         |
                         v
                   Final Caption
                         |
                         v
                  Download .txt
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Web application interface |
| PyTorch | Deep learning framework |
| BLIP | Image caption generation |
| Hugging Face Transformers | Pretrained model loading |
| MarianMT | English-to-Hindi translation |
| IndicTrans2 ONNX | English-to-Telugu translation |
| ONNX Runtime | Model inference |
| Pillow | Image processing |
| SentencePiece | Text tokenization |

## 📁 Project Structure

```text
AI_Image_Caption_Generator/
│
├── app.py
├── indictrans_telugu.py
├── requirements.txt
├── README.md
│
├── images/
│   └── test.jpeg
│
└── venv/                 # Local environment; do not share
```

## ⚙️ Installation and Setup

### Step 1: Install Python

Install Python 3.11 and verify:

```bash
python --version
```

### Step 2: Clone or download the project

Open the project folder in VS Code.

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

**Windows PowerShell:**

```powershell
.\ven