# 🧬 HelixAI — Advanced NLP Studio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Groq](https://img.shields.io/badge/AI-Groq%20LLaMA--3.3--70B-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **HelixAI** is a full-featured, production-ready NLP web application built with Streamlit.  
> It combines 10 powerful modules — from text cleaning and translation to AI summarisation and batch processing — all powered by **Groq LLaMA-3.3-70B** under the hood.

---
# ▶️ App Demo Link of the project (HelixAi)
https://helixai-atqi3jdfdkfskb8zufbkkt.streamlit.app/

## 🚀 Features

| Module | Capabilities |
|---|---|
| 📝 **Text Input & Cleaning** | HTML/URL/email stripping, stopword removal, lemmatisation, slang converter, regex builder, custom stopwords & dictionary |
| 🌐 **Translation & Language** | English ↔ Urdu / Pashto / Arabic / French / Spanish via deep-translator; auto language detection with probability chart |
| 🔍 **NLP Analysis** | POS tagging, keyword extraction, bigrams/trigrams, Named Entity Recognition with colour-coded entity cards |
| 💡 **Summarisation & SEO** | Extractive summariser, Groq AI summary, AI readability analysis, SEO score gauge with recommendations |
| 🔎 **Content Detection** | Toxicity score, sarcasm probability, VADER + TextBlob sentiment, AI-content estimator, duplicate sentence detector |
| 📤 **Extract & Parse** | Email / URL / phone / hashtag extraction with CSV export; emoji dictionary; AI FAQ generator |
| 📊 **Visualisations** | 30+ interactive Plotly charts: frequency bars, donut, treemap, sentiment gauge, radar, timeline, sunburst, box plot, network graph, WordCloud |
| 🤖 **AI Chatbot** | Multi-turn Groq LLaMA-3.3 chatbot with context awareness + smart auto-reply generator |
| 🎙 **Audio & Speech** | Whisper speech-to-text transcription; gTTS text-to-speech in 5 languages |
| 📦 **Batch Processing** | Upload multiple files; parallel NLP pipeline; downloadable CSV/XLSX report |

---

## 🏗️ Project Structure

```
nexalex/
├── app.py               ← Main Streamlit application (single-file)
├── requirements.txt     ← All Python dependencies
├── README.md            ← This file
└── .streamlit/
    └── secrets.toml     ← Local secrets file (⚠️ never commit to Git)
```

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nexalex-ai.git
cd nexalex-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key (local)

Create the secrets file:

```bash
mkdir -p .streamlit
```

Then create `.streamlit/secrets.toml` with the following content:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> 🔑 Get your free API key at [https://console.groq.com](https://console.groq.com)  
> ⚠️ **Never commit `secrets.toml` to Git.** Add it to `.gitignore`:
> ```
> .streamlit/secrets.toml
> ```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Cloud

### Step 1 — Push your code to GitHub

Make sure your repository contains:
- `app.py`
- `requirements.txt`
- `README.md`

**Do NOT include `.streamlit/secrets.toml`** — secrets are configured on the cloud dashboard instead.

### Step 2 — Create a new app on Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repository, branch, and set the **Main file path** to `app.py`
4. Click **"Deploy"**

### Step 3 — Add your GROQ_API_KEY secret

This is the most important step for AI features to work:

1. In your deployed app's dashboard, click the **⋮ (three-dot menu)** → **"Settings"**
2. Navigate to the **"Secrets"** tab
3. Paste the following into the secrets editor:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

4. Click **"Save"** — the app will automatically restart

> ✅ Once saved, all AI-powered features (Groq summarisation, chatbot, topic modelling, etc.) will be fully operational.

---

## 🔐 Security Notes

| Practice | Status |
|---|---|
| API key stored in `st.secrets` only | ✅ |
| No hardcoded credentials in source code | ✅ |
| `secrets.toml` excluded from Git | ✅ (add to `.gitignore`) |
| Graceful error message if key is missing | ✅ |

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit 1.32+ |
| **AI Backend** | Groq API · LLaMA-3.3-70B-Versatile |
| **NLP Core** | spaCy 3.7 (en_core_web_sm) · NLTK · TextBlob |
| **Visualisation** | Plotly · Matplotlib · WordCloud · NetworkX |
| **Translation** | deep-translator (Google Translate) · langdetect |
| **Speech** | OpenAI Whisper (STT) · gTTS (TTS) |
| **Data** | Pandas · NumPy · openpyxl |

---

## 📦 Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key from [console.groq.com](https://console.groq.com) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push and open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

<div align="center">
  <strong>🧬 NexaLex AI · Advanced NLP Studio · Groq LLaMA-3.3 · spaCy · Streamlit</strong>
</div>
