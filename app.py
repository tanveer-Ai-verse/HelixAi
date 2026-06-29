"""
🧬 NexaLex AI — Advanced NLP Studio
=====================================
Author  : NexaLex Team
Version : 2.0.0 (Streamlit Cloud Edition)
Backend : Groq LLaMA-3.3-70B-Versatile
Stack   : Streamlit · spaCy · NLTK · TextBlob · Plotly · WordCloud
"""

import streamlit as st
import pandas as pd
import re, html, json, time, os, io
import spacy
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import networkx as nx
import nltk
from groq import Groq

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexaLex AI | Advanced NLP Studio",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬",
)

# ─── Color Palette ─────────────────────────────────────────────────────────────
C1 = "#99CDD8"   # sky blue
C2 = "#DAEBE3"   # mint white
C3 = "#FDE8D3"   # soft peach
C4 = "#F3C3B2"   # dusty rose
C5 = "#CFD6C4"   # sage green
C6 = "#657166"   # deep slate


def rgba_c6(alpha: float = 0.27) -> str:
    return f"rgba(101,113,102,{alpha})"

def rgba_c1(alpha: float = 0.27) -> str:
    return f"rgba(153,205,216,{alpha})"

def rgba_c4(alpha: float = 0.27) -> str:
    return f"rgba(243,195,178,{alpha})"


GRID_COLOR  = rgba_c6(0.27)
LINE_COLOR  = rgba_c6(0.33)
CHART_COLORS = [C1, C4, C3, C5, "#8ecfd8", "#e8a49a", "#b5cdb8", "#a3c4c9"]

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [data-testid="stAppViewContainer"] {{
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1520 100%) !important;
    font-family: 'Space Grotesk', 'Exo 2', sans-serif !important;
    color: {C2} !important;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0d1520 0%, #111827 100%) !important;
    border-right: 1px solid {GRID_COLOR} !important;
}}
[data-testid="stSidebar"] * {{ color: {C2} !important; }}

h1 {{
    font-family: 'Orbitron', monospace !important;
    color: {C1} !important;
    font-size: 2.2rem !important;
    letter-spacing: 3px !important;
    text-shadow: 0 0 30px {rgba_c1(0.4)} !important;
}}
h2 {{
    font-family: 'Orbitron', monospace !important;
    color: {C4} !important;
    font-size: 1.4rem !important;
    letter-spacing: 2px !important;
}}

.hero-banner {{
    text-align: center;
    padding: 40px 20px 30px;
    background: linear-gradient(135deg, #0a0e1a 0%, #141b2d 50%, #0d1520 100%);
    border: 1px solid {rgba_c1(0.2)};
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0 0 60px {rgba_c1(0.07)};
}}

.neo-card {{
    background: linear-gradient(135deg, #141b2d, #1a2235);
    border: 1px solid {rgba_c1(0.2)};
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}}

.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 2px;
}}
.badge-blue  {{ background: {rgba_c1(0.2)};  border: 1px solid {rgba_c1(0.4)};  color: {C1}; }}
.badge-rose  {{ background: {rgba_c4(0.2)};  border: 1px solid {rgba_c4(0.4)};  color: {C4}; }}
.badge-sage  {{ background: rgba(207,214,196,0.2); border: 1px solid rgba(207,214,196,0.4); color: {C5}; }}
.badge-peach {{ background: rgba(253,232,211,0.2); border: 1px solid rgba(253,232,211,0.4); color: {C3}; }}

.section-header {{
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {C1};
    border-bottom: 1px solid {LINE_COLOR};
    padding-bottom: 10px;
    margin: 20px 0 15px;
}}

.result-box {{
    background: #0d1a2e;
    border: 1px solid {rgba_c1(0.27)};
    border-radius: 12px;
    padding: 15px 20px;
    font-family: 'Space Grotesk', monospace;
    font-size: 0.9rem;
    line-height: 1.7;
    color: {C2};
    max-height: 300px;
    overflow-y: auto;
}}

[data-testid="stTextArea"] textarea {{
    background: #0d1520 !important;
    border: 1px solid {LINE_COLOR} !important;
    border-radius: 12px !important;
    color: {C2} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
}}
[data-testid="stTextArea"] textarea:focus {{
    border-color: {C1} !important;
    box-shadow: 0 0 15px {rgba_c1(0.2)} !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, {C6} 0%, #3d4f52 100%) !important;
    color: {C1} !important;
    border: 1px solid {rgba_c1(0.4)} !important;
    border-radius: 10px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 2px !important;
    padding: 8px 20px !important;
    transition: all 0.3s !important;
    text-transform: uppercase !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {rgba_c1(0.27)} 0%, {rgba_c4(0.27)} 100%) !important;
    border-color: {C1} !important;
    box-shadow: 0 0 20px {rgba_c1(0.27)} !important;
    transform: translateY(-1px) !important;
}}

.stCheckbox > label, .stRadio > label, .stSelectbox > label, .stMultiSelect > label {{
    color: {C2} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important;
}}

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {{
    background: #0d1520 !important;
    border-color: {LINE_COLOR} !important;
    color: {C2} !important;
    border-radius: 10px !important;
}}

.stProgress > div > div {{
    background: linear-gradient(90deg, {C1}, {C4}) !important;
    border-radius: 10px !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: #141b2d !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {C5} !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 1px !important;
    padding: 6px 14px !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {rgba_c6(0.53)}, {rgba_c1(0.2)}) !important;
    color: {C1} !important;
    border: 1px solid {rgba_c1(0.27)} !important;
}}

.highlight-entity {{
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 600;
}}
.ent-PERSON  {{ background: {rgba_c4(0.27)}; border: 1px solid {rgba_c4(0.53)}; color: {C4}; }}
.ent-ORG     {{ background: {rgba_c1(0.27)}; border: 1px solid {rgba_c1(0.53)}; color: {C1}; }}
.ent-GPE     {{ background: rgba(253,232,211,0.27); border: 1px solid rgba(253,232,211,0.53); color: {C3}; }}
.ent-DATE    {{ background: rgba(207,214,196,0.27); border: 1px solid rgba(207,214,196,0.53); color: {C5}; }}
.ent-default {{ background: {rgba_c6(0.27)}; border: 1px solid {rgba_c6(0.53)}; color: {C2}; }}

div[data-testid="metric-container"] {{
    background: linear-gradient(135deg, #141b2d, #1a2235) !important;
    border: 1px solid {rgba_c1(0.27)} !important;
    border-radius: 12px !important;
    padding: 15px !important;
}}
div[data-testid="metric-container"] label {{
    color: {C5} !important;
    font-size: 0.78rem !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C1} !important;
    font-family: 'Orbitron', monospace !important;
}}

.warning-box {{
    background: {rgba_c4(0.07)};
    border: 1px solid {rgba_c4(0.4)};
    border-radius: 12px;
    padding: 12px 18px;
    color: {C4};
    font-size: 0.88rem;
}}
.success-box {{
    background: {rgba_c1(0.07)};
    border: 1px solid {rgba_c1(0.4)};
    border-radius: 12px;
    padding: 12px 18px;
    color: {C1};
    font-size: 0.88rem;
}}
.loading-pulse {{ animation: pulse 1.5s infinite; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: #0d1520; }}
::-webkit-scrollbar-thumb {{ background: {C6}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C1}; }}
</style>
""", unsafe_allow_html=True)


# ─── Groq AI Backend ───────────────────────────────────────────────────────────
def get_groq_client() -> Groq | None:
    """Return a Groq client using st.secrets, or None if key is missing."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except KeyError:
        return None


def run_ai_task(text: str, mode: str) -> str:
    """
    Core AI dispatch function.
    Modes: summary | keywords | topics | readability | qa
    Returns the model's text response, or a friendly error string.
    """
    client = get_groq_client()
    if client is None:
        return (
            "⚠️ **Groq API key not configured.**\n\n"
            "Add `GROQ_API_KEY` to your Streamlit secrets. "
            "See the README for step-by-step instructions."
        )

    prompts = {
        "summary": (
            f"You are an expert summariser. Summarise the following text in 3–5 clear, "
            f"informative sentences. Be concise and accurate:\n\n{text[:4000]}"
        ),
        "keywords": (
            f"Extract the 15 most important keywords/phrases from the text below. "
            f"Return ONLY a valid JSON object with a single key 'keywords' whose value "
            f"is a list of strings. No preamble, no markdown fences:\n\n{text[:4000]}"
        ),
        "topics": (
            f"Identify the main topics in the text below. "
            f"Return ONLY a valid JSON object with a single key 'topics', whose value is "
            f"a list of objects each with 'name' (string) and 'score' (float 0-1). "
            f"No preamble, no markdown fences:\n\n{text[:4000]}"
        ),
        "readability": (
            f"Analyse the readability, writing style, and overall tone of the text below. "
            f"Provide exactly 4 bullet-point observations:\n\n{text[:4000]}"
        ),
        "qa": text,
    }

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompts.get(mode, text)}],
            max_tokens=800,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API error: {e}"


# ─── NLTK / spaCy Setup ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP models…")
def load_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_resource(show_spinner=False)
def load_sia():
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)
    return SentimentIntensityAnalyzer()


try:
    nlp = load_nlp()
    sia = load_sia()
    NLP_OK = True
except Exception as _nlp_err:
    st.error(f"NLP load error: {_nlp_err}")
    NLP_OK = False

# ─── Session State ─────────────────────────────────────────────────────────────
_defaults = {
    "result": None, "raw_text": "", "processed_text": "",
    "history": [], "chat_messages": [], "batch_results": [],
    "current_text": "",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
  <h1>🧬 NEXALEX AI</h1>
  <p style="color:#CFD6C4;font-size:0.95rem;letter-spacing:2px;
            margin-top:8px;font-family:'Space Grotesk'">
    ADVANCED NLP STUDIO · MULTI-LANGUAGE · GROQ AI · REAL-TIME ANALYTICS
  </p>
  <div style="margin-top:12px">
    <span class="badge badge-blue">⚡ spaCy NLP</span>
    <span class="badge badge-rose">🌐 Multi-Language</span>
    <span class="badge badge-sage">📊 30+ Charts</span>
    <span class="badge badge-peach">🤖 Groq LLaMA-3.3</span>
    <span class="badge badge-blue">☁️ WordCloud</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-header'>🛠 MODULES</div>", unsafe_allow_html=True)
    module = st.radio("", [
        "📝 Text Input & Cleaning",
        "🌐 Translation & Language",
        "🔍 NLP Analysis",
        "💡 Summarization & SEO",
        "🔎 Content Detection",
        "📤 Extract & Parse",
        "📊 Visualizations",
        "🤖 AI Chatbot",
        "🎙 Audio & Speech",
        "📦 Batch Processing",
    ], label_visibility="collapsed")

    st.markdown("<div class='section-header'>⚙ CLEANING OPTIONS</div>", unsafe_allow_html=True)
    opt_html     = st.checkbox("Remove HTML Tags", True)
    opt_lower    = st.checkbox("Lowercase", True)
    opt_punct    = st.checkbox("Remove Punctuation", True)
    opt_num      = st.checkbox("Remove Numbers", False)
    opt_emoji    = st.checkbox("Remove Emojis", False)
    opt_url      = st.checkbox("Remove URLs", True)
    opt_email    = st.checkbox("Remove Emails", True)
    opt_sw       = st.checkbox("Remove Stopwords", True)
    opt_lemma    = st.checkbox("Lemmatization", True)
    opt_slang    = st.checkbox("Slang Converter", False)
    opt_markdown = st.checkbox("Clean Markdown", False)

    st.markdown("<div class='section-header'>🔠 LANGUAGE</div>", unsafe_allow_html=True)
    detect_lang = st.checkbox("Auto Detect Language", True)
    target_lang = st.selectbox(
        "Translate To",
        ["None", "Urdu (ur)", "Pashto (ps)", "Arabic (ar)", "French (fr)", "Spanish (es)"],
    )

    st.markdown("<div class='section-header'>📏 LIMITS</div>", unsafe_allow_html=True)
    max_chars = st.slider("Max Characters", 1000, 200000, 50000, 1000)

    st.markdown("<div class='section-header'>📊 VISUALIZATIONS</div>", unsafe_allow_html=True)
    show_wc      = st.checkbox("Word Cloud", True)
    show_network = st.checkbox("Co-occurrence Network", True)


# ─── Utility / NLP Functions ───────────────────────────────────────────────────
SLANG_DICT = {
    "u": "you", "r": "are", "ur": "your", "lol": "laughing out loud",
    "brb": "be right back", "omg": "oh my god", "tbh": "to be honest",
    "idk": "I don't know", "imo": "in my opinion", "btw": "by the way",
    "gr8": "great", "2": "to", "4": "for", "b4": "before", "cya": "see you",
    "thx": "thanks", "plz": "please", "ngl": "not gonna lie",
    "smh": "shaking my head", "fyi": "for your information",
    "afk": "away from keyboard", "gg": "good game",
    "dm": "direct message", "irl": "in real life", "np": "no problem",
}

TOXIC_WORDS = [
    "hate", "kill", "stupid", "idiot", "dumb", "loser", "trash", "garbage",
    "terrible", "awful", "horrible", "disgusting", "ugly", "pathetic",
]

STOP_WORDS_CUSTOM: set = set()


def clean_text(text: str, opts: dict) -> str:
    try:
        if opts.get("html"):     text = re.sub(r"<[^>]+>", "", html.unescape(text))
        if opts.get("url"):      text = re.sub(r"http\S+|www\S+", "", text)
        if opts.get("email"):    text = re.sub(r"\S+@\S+", "", text)
        if opts.get("lower"):    text = text.lower()
        if opts.get("punct"):    text = re.sub(r"[^\w\s]", "", text)
        if opts.get("num"):      text = re.sub(r"\d+", "", text)
        if opts.get("emoji"):    text = re.sub(r"[^\x00-\x7F]+", "", text)
        if opts.get("markdown"): text = re.sub(r"[#*`_~>\[\]()!]", "", text)
        if opts.get("slang"):
            text = " ".join(SLANG_DICT.get(w.lower(), w) for w in text.split())
        text = re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        st.error(f"Cleaning error: {e}")
    return text


@st.cache_data(show_spinner=False)
def process_text_nlp(text: str, sw: bool, lemma: bool, custom_sw_raw: str = ""):
    if not NLP_OK:
        return [], [], {}
    custom_sw = {w.strip().lower() for w in custom_sw_raw.split(",") if w.strip()}
    doc = nlp(text[:100_000])
    tokens, pos_data, entities = [], [], {}
    for token in doc:
        if token.is_space or not token.is_alpha:
            continue
        if sw and (token.is_stop or token.text.lower() in STOP_WORDS_CUSTOM or token.text.lower() in custom_sw):
            continue
        tokens.append(token.lemma_ if lemma else token.text)
        pos_data.append({"word": token.text, "pos": token.pos_,
                          "dep": token.dep_, "lemma": token.lemma_})
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    return tokens, pos_data, entities


@st.cache_data(show_spinner=False)
def full_pipeline_cached(text: str, sw: bool, lemma: bool) -> dict:
    doc = nlp(text[:100_000])
    tokens, pos_data, ner_data = [], [], []
    for tok in doc:
        if tok.is_space or not tok.is_alpha:
            continue
        if sw and tok.is_stop:
            continue
        tokens.append(tok.lemma_ if lemma else tok.text)
        pos_data.append({"word": tok.text, "pos": tok.pos_,
                          "dep": tok.dep_, "lemma": tok.lemma_})
    for ent in doc.ents:
        ner_data.append({"text": ent.text, "label": ent.label_})
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    sent_scores = []
    for s in sentences[:50]:
        sc = sia.polarity_scores(s)
        blob_sc = TextBlob(s).sentiment
        sent_scores.append({
            "sentence": s[:80], "compound": sc["compound"],
            "pos": sc["pos"], "neg": sc["neg"], "neu": sc["neu"],
            "subjectivity": round(blob_sc.subjectivity, 3),
        })
    overall = sia.polarity_scores(text[:5000])
    blob_overall = TextBlob(text[:5000]).sentiment
    bigrams  = Counter(" ".join(b) for b in zip(tokens, tokens[1:]))
    trigrams = Counter(" ".join(t) for t in zip(tokens, tokens[1:], tokens[2:]))
    return dict(
        tokens=tokens, pos_data=pos_data, ner_data=ner_data,
        sentences=sentences, sent_scores=sent_scores,
        overall_sentiment=overall,
        blob_polarity=round(blob_overall.polarity, 3),
        blob_subjectivity=round(blob_overall.subjectivity, 3),
        bigrams=bigrams, trigrams=trigrams,
    )


def get_sentences(text: str) -> list:
    if NLP_OK:
        doc = nlp(text[:50_000])
        return [sent.text.strip() for sent in doc.sents]
    return text.split(".")


def readability_score(text: str) -> tuple:
    sentences = get_sentences(text)
    words = text.split()
    if not sentences or not words:
        return 0, "N/A"
    avg_words = len(words) / max(len(sentences), 1)
    avg_syllables = sum(
        max(1, len(re.findall(r"[aeiouAEIOU]", w))) for w in words
    ) / max(len(words), 1)
    fk = 0.39 * avg_words + 11.8 * avg_syllables - 15.59
    fk = max(0, min(100, fk))
    grade = (
        "Very Hard" if fk < 30 else
        "Hard" if fk < 50 else
        "Standard" if fk < 65 else
        "Easy" if fk < 80 else "Very Easy"
    )
    return round(fk, 1), grade


def detect_sentiment(text: str) -> tuple:
    text_l = text.lower()
    pos_words = ["good", "great", "excellent", "amazing", "wonderful", "love", "happy",
                 "joy", "fantastic", "brilliant", "awesome", "best", "perfect",
                 "beautiful", "nice", "positive", "success"]
    neg_words = ["bad", "terrible", "awful", "hate", "horrible", "worst", "sad",
                 "angry", "disgusting", "pathetic", "fail", "wrong",
                 "broken", "poor", "negative", "disaster"]
    pos = sum(1 for w in pos_words if w in text_l)
    neg = sum(1 for w in neg_words if w in text_l)
    total = pos + neg
    if total == 0:
        return "Neutral", 50, pos, neg
    score = (pos / total) * 100
    label = "Positive" if score > 60 else ("Negative" if score < 40 else "Neutral")
    return label, round(score, 1), pos, neg


def detect_toxic(text: str) -> tuple:
    found = [w for w in TOXIC_WORDS if w in text.lower()]
    return min(100, len(found) * 12), found


def detect_sarcasm(text: str) -> int:
    signals = ["sure", "totally", "obviously", "clearly", "wow", "amazing",
               "really", "oh great"]
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    sig_count = sum(1 for s in signals if s in text.lower())
    return min(100, int(upper_ratio * 200 + sig_count * 10))


def extract_emails(text: str) -> list:
    return re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)

def extract_urls(text: str) -> list:
    return re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        text,
    )

def extract_phones(text: str) -> list:
    return re.findall(r"[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}", text)

def extract_hashtags(text: str) -> list:
    return re.findall(r"#\w+", text)


def extract_keywords(text: str, n: int = 10) -> list:
    if NLP_OK:
        doc = nlp(text[:50_000])
        words = [t.lemma_.lower() for t in doc
                 if t.is_alpha and not t.is_stop and len(t.text) > 2]
    else:
        words = [w for w in text.lower().split() if len(w) > 3]
    return Counter(words).most_common(n)


def get_bigrams(tokens: list) -> list:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

def get_trigrams(tokens: list) -> list:
    return [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]


def simple_summarize_extractive(text: str, n: int = 3) -> str:
    sentences = get_sentences(text)
    if len(sentences) <= n:
        return text
    freq = Counter(text.lower().split())
    scored = sorted(
        [(sum(freq.get(w.lower(), 0) for w in s.split()), s) for s in sentences],
        reverse=True,
    )
    return " ".join(s for _, s in scored[:n])


def detect_duplicates(text: str) -> list:
    sentences = get_sentences(text)
    seen, dups = set(), []
    for s in sentences:
        key = re.sub(r"\s+", " ", s.lower().strip())
        if key in seen and len(key) > 10:
            dups.append(s)
        seen.add(key)
    return dups


def emoji_meanings(text: str) -> dict:
    emoji_map = {
        "😀": "Grinning Face", "😂": "Face with Tears of Joy",
        "❤️": "Red Heart", "🔥": "Fire / Trending",
        "👍": "Thumbs Up / Approval", "😭": "Loudly Crying",
        "🎉": "Party / Celebration", "💯": "100% / Perfect",
        "🚀": "Rocket / Launch", "✨": "Sparkles / Magic",
        "😍": "Heart Eyes / Love", "🤔": "Thinking",
        "👀": "Eyes / Watching", "💪": "Muscle / Strong",
        "🙏": "Praying / Thank you", "🌟": "Star / Outstanding",
        "😎": "Cool / Sunglasses", "🎯": "Target / Accurate",
    }
    return {e: m for e, m in emoji_map.items() if e in text}


def faq_generator(text: str) -> list:
    sentences = get_sentences(text)
    faqs = []
    for sent in sentences[:5]:
        words = sent.split()
        if len(words) < 4:
            continue
        if any(w in sent.lower() for w in ["is", "are", "was", "were", "do", "does", "can"]):
            faqs.append(("Q: " + sent.strip() + "?", "A: " + sent.strip()))
        elif len(words) >= 5:
            faqs.append(
                ("Q: What is " + words[0].lower() + "?", "A: " + sent.strip())
            )
    return faqs[:5]


def seo_check(text: str) -> tuple:
    words = text.split()
    wc = len(words)
    sentences = get_sentences(text)
    kws = extract_keywords(text, 5)
    score, notes = 0, []
    if wc >= 300:
        score += 25
        notes.append(("✅", "Good word count"))
    else:
        notes.append(("⚠️", f"Low word count ({wc}). Aim for 300+"))
    avg_len = wc / max(len(sentences), 1)
    if avg_len < 20:
        score += 20
        notes.append(("✅", "Good sentence length"))
    else:
        notes.append(("⚠️", "Sentences too long for SEO"))
    if kws:
        score += 20
        notes.append(("✅", "Keywords found"))
    if wc >= 600:
        score += 15
        notes.append(("✅", "Long-form content bonus"))
    return min(100, score + 20), notes, kws


def wordcloud_img(tokens: list):
    if not tokens:
        return None
    freq = Counter(tokens)
    wc = WordCloud(
        width=900, height=380,
        background_color="#0d1520",
        colormap="GnBu",
        max_words=120,
        prefer_horizontal=0.8,
        relative_scaling=0.5,
    ).generate_from_frequencies(freq)
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#0d1520")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0d1520")
    plt.close(fig)
    buf.seek(0)
    return buf


def cooccurrence_network(tokens: list, top_n: int = 40, window: int = 3):
    tokens_top = [t for t, _ in Counter(tokens).most_common(top_n)]
    tok_set = set(tokens_top)
    G = nx.Graph()
    G.add_nodes_from(tokens_top)
    for i, t in enumerate(tokens):
        if t in tok_set:
            for j in range(i + 1, min(i + window + 1, len(tokens))):
                nb = tokens[j]
                if nb in tok_set and nb != t:
                    if G.has_edge(t, nb):
                        G[t][nb]["weight"] = G[t][nb].get("weight", 1) + 1
                    else:
                        G.add_edge(t, nb, weight=1)
    return G


def render_network_plotly(G):
    if len(G.nodes) < 3:
        return None
    pos = nx.spring_layout(G, k=0.8, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.7, color=rgba_c6(0.33)), hoverinfo="none",
    )
    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]
    node_degree = [G.degree(n) for n in G.nodes]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=list(G.nodes), textposition="top center",
        textfont=dict(size=9, color=C2),
        marker=dict(
            size=[6 + d * 1.5 for d in node_degree],
            color=node_degree,
            colorscale=[[0, C6], [0.5, C1], [1, C4]],
            showscale=True,
            colorbar=dict(title="Degree", thickness=10, tickfont=dict(color=C2)),
            line=dict(color=C4, width=0.8),
        ),
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1520",
        font=dict(family="Space Grotesk", color=C2),
        title=dict(text="Word Co-occurrence Network",
                   font=dict(color=C1, size=14)),
        showlegend=False, height=480,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def chart_layout(fig, title: str = ""):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=14, color=C2)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,21,32,0.8)",
        font=dict(family="Space Grotesk", color=C2, size=11),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=LINE_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=LINE_COLOR),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — Text Input & Cleaning
# ═══════════════════════════════════════════════════════════════════════════════
if "📝 Text Input & Cleaning" in module:
    st.markdown("<div class='section-header'>📝 TEXT INPUT & CLEANING</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["✍ Paste Text", "📁 Upload File", "📦 Multi-File"])

    with tab1:
        raw = st.text_area("Enter Text", height=200,
                           placeholder="Paste your text here…", key="input_text")
        if raw:
            st.session_state.raw_text = raw
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("🎲 Load Sample Text"):
                st.session_state.raw_text = (
                    "Artificial intelligence is transforming the world at an unprecedented pace. "
                    "Companies like OpenAI, Google, and Meta are investing billions in large language models. "
                    "In 2023, ChatGPT reached 100 million users in just two months. "
                    "Critics argue that AI poses existential risks, while proponents believe it will solve "
                    "humanity's greatest challenges — from climate change to cancer research."
                )

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            f = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
            if f:
                if f.name.endswith(".txt"):
                    st.session_state.raw_text = f.read().decode()
                else:
                    df = pd.read_csv(f)
                    st.session_state.raw_text = " ".join(df.astype(str).values.flatten())
                st.success(f"✅ Loaded: {f.name}")
        with col_b:
            if st.session_state.raw_text:
                st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
                st.write(f"**Characters:** {len(st.session_state.raw_text):,}")
                st.write(f"**Words:** {len(st.session_state.raw_text.split()):,}")
                st.write(f"**Lines:** {st.session_state.raw_text.count(chr(10)):,}")
                st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        files = st.file_uploader("Upload Multiple Files", type=["txt", "csv"],
                                 accept_multiple_files=True)
        if files:
            combined = []
            for mf in files:
                txt = mf.read().decode()
                combined.append(txt)
                st.session_state.batch_results.append({"file": mf.name, "text": txt})
            st.session_state.raw_text = "\n\n".join(combined)
            st.success(f"✅ {len(files)} files loaded and combined")

    st.markdown("<div class='section-header'>🔧 REGEX RULE BUILDER</div>", unsafe_allow_html=True)
    col_rx1, col_rx2 = st.columns([2, 1])
    with col_rx1:
        regex_pattern = st.text_input("Custom Regex Pattern",
                                      placeholder=r"e.g. \d{4}-\d{2}-\d{2}")
        regex_replace = st.text_input("Replace With",
                                      placeholder="(leave empty to remove)")
    with col_rx2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ Apply Regex") and regex_pattern and st.session_state.raw_text:
            try:
                st.session_state.raw_text = re.sub(
                    regex_pattern, regex_replace or "", st.session_state.raw_text
                )
                st.success("✅ Regex applied!")
            except Exception as e:
                st.error(f"Regex error: {e}")

    st.markdown("<div class='section-header'>📚 CUSTOM STOPWORDS</div>", unsafe_allow_html=True)
    col_sw1, col_sw2 = st.columns(2)
    with col_sw1:
        custom_sw = st.text_input("Add Custom Stopwords (comma-separated)",
                                  placeholder="e.g. also, however")
        if custom_sw:
            STOP_WORDS_CUSTOM.update(w.strip().lower() for w in custom_sw.split(","))
            st.success(f"✅ {len(STOP_WORDS_CUSTOM)} custom stopwords active")
    with col_sw2:
        custom_dict = st.text_area(
            "Custom Dictionary (word=replacement, one per line)",
            height=80, placeholder="colour=color\nbehaviour=behavior",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn2, _ = st.columns([1, 2, 1])
    with col_btn2:
        run_clean = st.button("🚀 PROCESS & ANALYZE", use_container_width=True)

    opts = {
        "html": opt_html, "lower": opt_lower, "punct": opt_punct,
        "num": opt_num, "emoji": opt_emoji, "url": opt_url,
        "email": opt_email, "slang": opt_slang, "markdown": opt_markdown,
    }

    if run_clean and st.session_state.raw_text:
        text = st.session_state.raw_text[:max_chars]
        if custom_dict:
            for line in custom_dict.strip().split("\n"):
                if "=" in line:
                    old, new = line.split("=", 1)
                    text = text.replace(old.strip(), new.strip())
        with st.spinner("⚡ Running NLP pipeline…"):
            cleaned = clean_text(text, opts)
            tokens, pos_data, entities = process_text_nlp(cleaned, opt_sw, opt_lemma)
            final = " ".join(tokens)
            st.session_state.result = (cleaned, tokens, final, pos_data, entities, text)
            st.session_state.processed_text = final
            st.session_state.current_text = text

    if st.session_state.result:
        cleaned, tokens, final, pos_data, entities, orig_text = st.session_state.result
        st.markdown("<div class='section-header'>📊 STATISTICS</div>", unsafe_allow_html=True)
        orig_words = len(orig_text.split())
        proc_words = len(tokens)
        reduction  = round((1 - proc_words / max(orig_words, 1)) * 100, 1)
        rs, rg = readability_score(cleaned)
        sent_label, sent_score, pos_c, neg_c = detect_sentiment(cleaned)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Original Words",  f"{orig_words:,}")
        m2.metric("Processed Words", f"{proc_words:,}")
        m3.metric("Reduction",       f"{reduction}%")
        m4.metric("Readability",     f"{rs} ({rg})")
        m5.metric("Sentiment",       sent_label)
        m6.metric("Sentences",       len(get_sentences(cleaned)))

        st.markdown("<div class='section-header'>✅ CLEANED TEXT</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'>{final}</div>", unsafe_allow_html=True)

        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            st.download_button("⬇ Download .txt", final, "cleaned.txt",
                               "text/plain", use_container_width=True)
        with col_dl2:
            df_out = pd.DataFrame({"token": tokens})
            st.download_button("⬇ Download .csv", df_out.to_csv(index=False),
                               "tokens.csv", "text/csv", use_container_width=True)
        with col_dl3:
            st.download_button("⬇ Download .md", final, "cleaned.md",
                               "text/markdown", use_container_width=True)

        if tokens:
            freq = Counter(tokens)
            top  = dict(freq.most_common(15))
            fig = go.Figure(go.Bar(
                x=list(top.values()), y=list(top.keys()),
                orientation="h",
                marker=dict(color=list(range(len(top))),
                            colorscale=[[0, C6], [0.5, C1], [1, C4]], showscale=False),
                text=list(top.values()), textposition="outside",
            ))
            chart_layout(fig, "🔤 Word Frequency")
            fig.update_layout(height=420,
                              yaxis=dict(autorange="reversed",
                                        gridcolor=GRID_COLOR, linecolor=LINE_COLOR))
            st.plotly_chart(fig, use_container_width=True)

        if show_wc and tokens:
            st.markdown("<div class='section-header'>☁️ WORD CLOUD</div>", unsafe_allow_html=True)
            wc_buf = wordcloud_img(tokens)
            if wc_buf:
                st.image(wc_buf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — Translation & Language
# ═══════════════════════════════════════════════════════════════════════════════
elif "🌐 Translation" in module:
    st.markdown("<div class='section-header'>🌐 TRANSLATION & LANGUAGE DETECTION</div>",
                unsafe_allow_html=True)
    input_text = st.text_area("Text to Translate", height=180,
                              value=st.session_state.raw_text or "")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        direction = st.selectbox("Translation Direction", [
            "English → Urdu", "Urdu → English",
            "English → Pashto", "English → Arabic", "Arabic → English",
            "English → French", "English → Spanish",
        ])
    with col_t2:
        st.markdown("<br>", unsafe_allow_html=True)
        do_translate = st.button("🌐 Translate", use_container_width=True)

    if do_translate and input_text:
        with st.spinner("🌐 Translating…"):
            try:
                from deep_translator import GoogleTranslator
                lang_map = {
                    "English → Urdu":   ("en", "ur"), "Urdu → English":   ("ur", "en"),
                    "English → Pashto": ("en", "ps"), "English → Arabic": ("en", "ar"),
                    "Arabic → English": ("ar", "en"), "English → French": ("en", "fr"),
                    "English → Spanish": ("en", "es"),
                }
                src, tgt = lang_map[direction]
                translated = GoogleTranslator(source=src, target=tgt).translate(input_text[:5000])
                st.markdown("<div class='section-header'>📝 TRANSLATION RESULT</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='result-box'>{translated}</div>",
                            unsafe_allow_html=True)
                st.download_button("⬇ Download Translation", translated, "translation.txt")
            except ImportError:
                st.warning("Install: `pip install deep-translator`")
            except Exception as e:
                st.error(f"Translation error: {e}")

    st.markdown("<div class='section-header'>🔍 AUTO LANGUAGE DETECTION</div>",
                unsafe_allow_html=True)
    detect_input = st.text_area("Text for Detection", height=100)
    if st.button("🔍 Detect Language"):
        try:
            from langdetect import detect, detect_langs
            lang  = detect(detect_input)
            probs = detect_langs(detect_input)
            st.success(f"Detected Language: **{lang.upper()}**")
            lang_data = {
                str(p).split(":")[0]: float(str(p).split(":")[1]) for p in probs
            }
            fig = go.Figure(go.Bar(
                x=list(lang_data.keys()), y=list(lang_data.values()),
                marker_color=[C1, C4, C3, C5][: len(lang_data)],
            ))
            chart_layout(fig, "Language Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install `langdetect`: `pip install langdetect`")
        except Exception as e:
            st.error(f"Detection error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — NLP Analysis
# ═══════════════════════════════════════════════════════════════════════════════
elif "🔍 NLP Analysis" in module:
    st.markdown("<div class='section-header'>🔍 NLP ANALYSIS ENGINE</div>", unsafe_allow_html=True)
    text_input = st.text_area("Text to Analyze", height=180,
                              value=st.session_state.processed_text or st.session_state.raw_text or "")
    _, col_run2 = st.columns([3, 1])
    with col_run2:
        run_nlp = st.button("⚡ Analyze", use_container_width=True)

    if run_nlp and text_input:
        with st.spinner("Running NLP pipeline…"):
            tokens, pos_data, entities = process_text_nlp(text_input, False, True)
            kw       = extract_keywords(text_input, 15)
            bigrams  = get_bigrams(tokens)
            trigrams = get_trigrams(tokens)

        nlp_tab1, nlp_tab2, nlp_tab3, nlp_tab4 = st.tabs(
            ["🏷 POS Tags", "🔑 Keywords", "🔗 N-Grams", "🏢 Entities"]
        )

        with nlp_tab1:
            st.markdown("<div class='section-header'>PART-OF-SPEECH TAGGING</div>",
                        unsafe_allow_html=True)
            if pos_data:
                df_pos = pd.DataFrame(pos_data[:50])
                st.dataframe(df_pos, use_container_width=True, height=300)
                pos_counts = Counter(d["pos"] for d in pos_data)
                fig = go.Figure(go.Pie(
                    labels=list(pos_counts.keys()), values=list(pos_counts.values()),
                    hole=0.45, marker=dict(colors=CHART_COLORS),
                ))
                chart_layout(fig, "POS Tag Distribution")
                st.plotly_chart(fig, use_container_width=True)

        with nlp_tab2:
            st.markdown("<div class='section-header'>KEYWORD EXTRACTION</div>",
                        unsafe_allow_html=True)
            if kw:
                kw_words  = [k[0] for k in kw]
                kw_scores = [k[1] for k in kw]
                col_kw1, col_kw2 = st.columns(2)
                with col_kw1:
                    fig = go.Figure(go.Bar(
                        x=kw_scores, y=kw_words, orientation="h",
                        marker=dict(color=kw_scores,
                                    colorscale=[[0, C6], [1, C1]]),
                    ))
                    chart_layout(fig, "Keyword Importance")
                    fig.update_layout(yaxis=dict(autorange="reversed",
                                                 gridcolor=GRID_COLOR))
                    st.plotly_chart(fig, use_container_width=True)
                with col_kw2:
                    fig = go.Figure(go.Treemap(
                        labels=kw_words, parents=[""] * len(kw_words),
                        values=kw_scores,
                        marker=dict(colorscale=[[0, C6], [0.5, C1], [1, C4]]),
                    ))
                    chart_layout(fig, "Keyword Treemap")
                    st.plotly_chart(fig, use_container_width=True)

        with nlp_tab3:
            st.markdown("<div class='section-header'>N-GRAM ANALYSIS</div>",
                        unsafe_allow_html=True)
            col_bi, col_tri = st.columns(2)
            with col_bi:
                if bigrams:
                    bi_freq = Counter(f"{a} {b}" for a, b in bigrams).most_common(10)
                    fig = go.Figure(go.Bar(
                        x=[v for _, v in bi_freq], y=[k for k, _ in bi_freq],
                        orientation="h", marker_color=C1,
                    ))
                    chart_layout(fig, "Top Bigrams")
                    fig.update_layout(
                        yaxis=dict(autorange="reversed", gridcolor=GRID_COLOR), height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with col_tri:
                if trigrams:
                    tri_freq = Counter(f"{a} {b} {c}" for a, b, c in trigrams).most_common(8)
                    fig = go.Figure(go.Bar(
                        x=[v for _, v in tri_freq], y=[k for k, _ in tri_freq],
                        orientation="h", marker_color=C4,
                    ))
                    chart_layout(fig, "Top Trigrams")
                    fig.update_layout(
                        yaxis=dict(autorange="reversed", gridcolor=GRID_COLOR), height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with nlp_tab4:
            st.markdown("<div class='section-header'>NAMED ENTITY RECOGNITION</div>",
                        unsafe_allow_html=True)
            if entities:
                color_map = {
                    "PERSON": "ent-PERSON", "ORG": "ent-ORG",
                    "GPE": "ent-GPE", "DATE": "ent-DATE", "TIME": "ent-DATE",
                }
                ent_html = ""
                for label, items in entities.items():
                    css = color_map.get(label, "ent-default")
                    ent_html += (
                        f"<div style='margin:8px 0'>"
                        f"<strong style='color:{C5};font-size:0.8rem'>{label}:</strong> "
                    )
                    for item in set(items[:8]):
                        ent_html += f"<span class='highlight-entity {css}'>{item}</span> "
                    ent_html += "</div>"
                st.markdown(f"<div class='neo-card'>{ent_html}</div>",
                            unsafe_allow_html=True)
                ent_counts = {k: len(v) for k, v in entities.items()}
                fig = go.Figure(go.Bar(
                    x=list(ent_counts.keys()), y=list(ent_counts.values()),
                    marker=dict(color=CHART_COLORS[: len(ent_counts)]),
                ))
                chart_layout(fig, "Entity Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No named entities detected.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — Summarization & SEO
# ═══════════════════════════════════════════════════════════════════════════════
elif "💡 Summarization" in module:
    st.markdown("<div class='section-header'>💡 SUMMARIZATION & SEO OPTIMIZER</div>",
                unsafe_allow_html=True)
    text_s = st.text_area("Input Text", height=200,
                          value=st.session_state.raw_text or "")

    s_tab1, s_tab2, s_tab3, s_tab4 = st.tabs(
        ["📋 Extractive", "🤖 Groq AI Summary", "📝 AI Readability", "📈 SEO"]
    )

    with s_tab1:
        n_sents = st.slider("Number of sentences", 1, 10, 3)
        if st.button("⚡ Extractive Summarize"):
            if text_s:
                with st.spinner("Summarizing…"):
                    summary = simple_summarize_extractive(text_s, n_sents)
                st.markdown(f"<div class='result-box'>{summary}</div>",
                            unsafe_allow_html=True)
                orig_len = len(text_s.split())
                summ_len = len(summary.split())
                c1_, c2_, c3_ = st.columns(3)
                c1_.metric("Original Words", orig_len)
                c2_.metric("Summary Words", summ_len)
                c3_.metric("Compression",
                           f"{round((1 - summ_len / max(orig_len, 1)) * 100, 1)}%")

    with s_tab2:
        if st.button("🤖 Groq AI Summary"):
            if text_s:
                with st.spinner("Generating Groq AI summary…"):
                    result = run_ai_task(text_s, "summary")
                st.markdown(f"<div class='result-box'>{result}</div>",
                            unsafe_allow_html=True)

    with s_tab3:
        if st.button("📝 Groq Readability Analysis"):
            if text_s:
                with st.spinner("Analyzing…"):
                    result = run_ai_task(text_s, "readability")
                st.markdown(f"<div class='result-box'>{result}</div>",
                            unsafe_allow_html=True)

    with s_tab4:
        if st.button("📈 Run SEO Check"):
            if text_s:
                seo_score, seo_notes, seo_kws = seo_check(text_s)
                col_seo1, col_seo2 = st.columns([1, 2])
                with col_seo1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=seo_score,
                        title={"text": "SEO Score",
                               "font": {"color": C2, "family": "Space Grotesk"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": C6},
                            "bar": {"color": C1},
                            "steps": [
                                {"range": [0, 40],  "color": rgba_c4(0.27)},
                                {"range": [40, 70], "color": "rgba(253,232,211,0.27)"},
                                {"range": [70, 100], "color": rgba_c1(0.27)},
                            ],
                            "threshold": {
                                "line": {"color": C4, "width": 3}, "value": seo_score
                            },
                        },
                    ))
                    chart_layout(fig)
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)
                with col_seo2:
                    for icon, note in seo_notes:
                        color = C1 if "✅" in icon else C4
                        st.markdown(
                            f"<div style='padding:6px;color:{color};font-size:0.88rem'>"
                            f"{icon} {note}</div>",
                            unsafe_allow_html=True,
                        )
                    for kw_word, freq in seo_kws:
                        st.markdown(
                            f"<span class='badge badge-blue'>{kw_word} ({freq})</span>",
                            unsafe_allow_html=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — Content Detection
# ═══════════════════════════════════════════════════════════════════════════════
elif "🔎 Content Detection" in module:
    st.markdown("<div class='section-header'>🔎 CONTENT DETECTION & ANALYSIS</div>",
                unsafe_allow_html=True)
    text_d = st.text_area("Text to Analyze", height=200,
                          value=st.session_state.raw_text or "")

    if st.button("🔍 Run Full Detection", use_container_width=True) and text_d:
        d_tab1, d_tab2, d_tab3, d_tab4, d_tab5 = st.tabs(
            ["😤 Toxicity", "😏 Sarcasm", "😊 Sentiment", "🤖 AI Content", "🔁 Duplicates"]
        )

        with d_tab1:
            tox_score, tox_words = detect_toxic(text_d)
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=tox_score,
                    title={"text": "Toxicity Score", "font": {"color": C2}},
                    delta={"reference": 50},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": C4 if tox_score > 50 else C1},
                        "steps": [
                            {"range": [0, 30],  "color": rgba_c1(0.2)},
                            {"range": [30, 70], "color": "rgba(253,232,211,0.2)"},
                            {"range": [70, 100], "color": rgba_c4(0.2)},
                        ],
                    },
                ))
                chart_layout(fig)
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            with col_t2:
                if tox_words:
                    st.warning(f"⚠️ Detected {len(tox_words)} flagged word(s)")
                    for w in tox_words:
                        st.markdown(
                            f"<span class='badge badge-rose'>❌ {w}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("✅ No toxic content detected")

        with d_tab2:
            sarc_score = detect_sarcasm(text_d)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=sarc_score,
                title={"text": "Sarcasm Probability %", "font": {"color": C2}},
                gauge={
                    "axis": {"range": [0, 100]}, "bar": {"color": C3},
                    "steps": [
                        {"range": [0, 40],  "color": "rgba(207,214,196,0.2)"},
                        {"range": [40, 70], "color": "rgba(253,232,211,0.2)"},
                        {"range": [70, 100], "color": rgba_c4(0.2)},
                    ],
                },
            ))
            chart_layout(fig)
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        with d_tab3:
            sent_label, sent_score, pos_c, neg_c = detect_sentiment(text_d)
            sc_nltk = sia.polarity_scores(text_d[:5000])
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Sentiment Label", sent_label)
                st.metric("VADER Compound", f"{sc_nltk['compound']:.3f}")
                st.metric("TextBlob Polarity",
                          f"{TextBlob(text_d[:5000]).sentiment.polarity:.3f}")
            with col_s2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=sent_score,
                    title={"text": "Sentiment Score", "font": {"color": C2}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": C1 if sent_score > 50 else C4},
                        "steps": [
                            {"range": [0, 40],  "color": rgba_c4(0.2)},
                            {"range": [40, 70], "color": "rgba(207,214,196,0.2)"},
                            {"range": [60, 100], "color": rgba_c1(0.2)},
                        ],
                    },
                ))
                chart_layout(fig)
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

        with d_tab4:
            words = text_d.split()
            avg_wl = np.mean([len(w) for w in words]) if words else 0
            sentences = get_sentences(text_d)
            avg_sl = len(words) / max(len(sentences), 1)
            repetition = len(words) - len({w.lower() for w in words})
            ai_score = min(100, int(avg_wl * 5 + avg_sl * 2 + repetition * 0.5))
            st.metric("AI Content Probability", f"{ai_score}%")
            fig = go.Figure(go.Bar(
                x=["Avg Word Length", "Avg Sentence Length", "Repetition"],
                y=[avg_wl, avg_sl, repetition],
                marker_color=[C1, C3, C4],
            ))
            chart_layout(fig, "AI Signal Analysis")
            st.plotly_chart(fig, use_container_width=True)

        with d_tab5:
            dups = detect_duplicates(text_d)
            if dups:
                st.warning(f"⚠️ {len(dups)} duplicate sentence(s) found:")
                for d in dups:
                    st.markdown(
                        f"<div class='warning-box'>🔁 {d[:150]}…</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ No duplicate sentences detected")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 6 — Extract & Parse
# ═══════════════════════════════════════════════════════════════════════════════
elif "📤 Extract" in module:
    st.markdown("<div class='section-header'>📤 EXTRACTION & PARSING ENGINE</div>",
                unsafe_allow_html=True)
    text_e = st.text_area("Input Text", height=180,
                          value=st.session_state.raw_text or "")

    if st.button("⚡ Extract All", use_container_width=True) and text_e:
        emails   = extract_emails(text_e)
        urls     = extract_urls(text_e)
        phones   = extract_phones(text_e)
        hashtags = extract_hashtags(text_e)
        emojis   = emoji_meanings(text_e)

        e_tab1, e_tab2, e_tab3, e_tab4, e_tab5 = st.tabs(
            ["📧 Emails", "🔗 URLs", "📞 Phones", "#️⃣ Hashtags", "😀 Emojis"]
        )

        with e_tab1:
            st.metric("Emails Found", len(emails))
            for e in emails:
                st.markdown(
                    f"<span class='badge badge-blue'>📧 {e}</span>",
                    unsafe_allow_html=True,
                )
            if emails:
                st.download_button(
                    "⬇ Download Emails",
                    pd.DataFrame({"Email": emails}).to_csv(index=False),
                    "emails.csv",
                )

        with e_tab2:
            st.metric("URLs Found", len(urls))
            for u in urls:
                st.markdown(
                    f"<div class='result-box' style='margin:4px 0;max-height:60px'>{u}</div>",
                    unsafe_allow_html=True,
                )

        with e_tab3:
            st.metric("Phone Numbers Found", len(phones))
            for p in phones:
                st.markdown(
                    f"<span class='badge badge-sage'>📞 {p}</span>",
                    unsafe_allow_html=True,
                )

        with e_tab4:
            st.metric("Hashtags Found", len(hashtags))
            for tag in hashtags:
                st.markdown(
                    f"<span class='badge badge-rose'>{tag}</span>",
                    unsafe_allow_html=True,
                )
            if hashtags:
                ht_freq = Counter(hashtags)
                fig = go.Figure(go.Bar(
                    x=list(ht_freq.keys()), y=list(ht_freq.values()),
                    marker_color=C4,
                ))
                chart_layout(fig, "Hashtag Frequency")
                st.plotly_chart(fig, use_container_width=True)

        with e_tab5:
            if emojis:
                for emoji_char, meaning in emojis.items():
                    st.markdown(
                        f"<div class='neo-card' style='padding:10px 15px'>"
                        f"<span style='font-size:1.5rem'>{emoji_char}</span> "
                        f"<span style='color:{C2};'>{meaning}</span></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No recognized emojis found")

    st.markdown("<div class='section-header'>❓ FAQ GENERATOR</div>",
                unsafe_allow_html=True)
    if st.button("🤖 Generate FAQs") and text_e:
        faqs = faq_generator(text_e)
        for q, a in faqs:
            with st.expander(q):
                st.write(a)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 7 — Visualizations
# ═══════════════════════════════════════════════════════════════════════════════
elif "📊 Visualizations" in module:
    st.markdown("<div class='section-header'>📊 ADVANCED VISUALIZATIONS</div>",
                unsafe_allow_html=True)
    text_v = st.text_area(
        "Text for Visualization", height=150,
        value=st.session_state.raw_text or st.session_state.processed_text or "",
    )

    if st.button("📊 Generate All Charts", use_container_width=True) and text_v and NLP_OK:
        with st.spinner("Generating visualizations…"):
            tokens, pos_data, entities = process_text_nlp(text_v, True, True)
            freq = Counter(tokens)
            kw = extract_keywords(text_v, 12)
            sent_label, sent_score, pos_c, neg_c = detect_sentiment(text_v)
            sentences  = get_sentences(text_v)
            words      = text_v.split()
            rs, rg     = readability_score(text_v)
            sarc       = detect_sarcasm(text_v)
            tox_score, _ = detect_toxic(text_v)
            R = full_pipeline_cached(text_v, True, True)

        v_tab1, v_tab2, v_tab3, v_tab4, v_tab5, v_tab6 = st.tabs(
            ["📈 Frequency", "🎭 Sentiment", "📐 Structure",
             "🌐 Network", "☁️ WordCloud", "🤖 AI Topics"]
        )

        with v_tab1:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                top = dict(freq.most_common(12))
                fig = go.Figure(go.Bar(
                    x=list(top.keys()), y=list(top.values()),
                    marker=dict(color=list(range(len(top))),
                                colorscale=[[0, C6], [0.5, C1], [1, C4]]),
                ))
                chart_layout(fig, "Word Frequency")
                st.plotly_chart(fig, use_container_width=True)
            with col_v2:
                if kw:
                    fig = go.Figure(go.Pie(
                        labels=[k[0] for k in kw[:8]], values=[k[1] for k in kw[:8]],
                        hole=0.5, marker=dict(colors=CHART_COLORS),
                    ))
                    chart_layout(fig, "Keyword Distribution (Donut)")
                    st.plotly_chart(fig, use_container_width=True)
            col_v3, col_v4 = st.columns(2)
            with col_v3:
                if kw:
                    fig = go.Figure(go.Treemap(
                        labels=[k[0] for k in kw], parents=[""] * len(kw),
                        values=[k[1] for k in kw],
                        marker=dict(colorscale=[[0, C6], [0.5, C1], [1, C4]]),
                    ))
                    chart_layout(fig, "Keyword Treemap")
                    st.plotly_chart(fig, use_container_width=True)
            with col_v4:
                sent_lengths = [len(s.split()) for s in sentences if s.strip()]
                if sent_lengths:
                    fig = go.Figure(go.Histogram(
                        x=sent_lengths, nbinsx=15,
                        marker=dict(color=C1, line=dict(color=C6, width=1)),
                    ))
                    chart_layout(fig, "Sentence Length Distribution")
                    st.plotly_chart(fig, use_container_width=True)

        with v_tab2:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=sent_score,
                    title={"text": "Sentiment Score",
                           "font": {"color": C2, "family": "Space Grotesk"}},
                    delta={"reference": 50,
                           "increasing": {"color": C1}, "decreasing": {"color": C4}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": C6},
                        "bar": {"color": C1 if sent_score > 50 else C4, "thickness": 0.25},
                        "bgcolor": "#0d1520",
                        "steps": [
                            {"range": [0, 30],  "color": rgba_c4(0.2)},
                            {"range": [30, 60], "color": "rgba(207,214,196,0.2)"},
                            {"range": [60, 100], "color": rgba_c1(0.2)},
                        ],
                        "threshold": {"line": {"color": C3, "width": 4},
                                      "value": sent_score},
                    },
                ))
                chart_layout(fig, "Sentiment Gauge")
                st.plotly_chart(fig, use_container_width=True)
            with col_s2:
                categories = ["Sentiment", "Readability", "Toxicity-Free",
                              "Length Score", "Keyword Rich"]
                values = [
                    sent_score, min(100, rs * 1.5), 100 - tox_score,
                    min(100, len(words) / 5), min(100, len(kw) * 7),
                ]
                fig = go.Figure(go.Scatterpolar(
                    r=values + [values[0]], theta=categories + [categories[0]],
                    fill="toself", fillcolor=rgba_c1(0.2),
                    line=dict(color=C1, width=2), name="Text Quality",
                ))
                fig.update_layout(polar=dict(
                    bgcolor="#0d1520",
                    radialaxis=dict(range=[0, 100], color=C5),
                    angularaxis=dict(color=C2),
                ))
                chart_layout(fig, "Radar: Text Quality")
                st.plotly_chart(fig, use_container_width=True)

            if R.get("sent_scores"):
                st.markdown("<div class='section-header'>Sentiment Timeline</div>",
                            unsafe_allow_html=True)
                df_sent = pd.DataFrame(R["sent_scores"])
                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(
                    x=list(range(len(df_sent))), y=df_sent["compound"],
                    mode="lines+markers",
                    line=dict(color=C1, width=2),
                    marker=dict(
                        color=df_sent["compound"],
                        colorscale=[[0, C4], [0.5, C5], [1, C1]], size=8,
                        showscale=False,
                    ),
                    fill="tozeroy", fillcolor=rgba_c1(0.05),
                    hovertemplate="<b>Sentence %{x}</b><br>Score: %{y:.3f}<extra></extra>",
                ))
                fig_tl.add_hline(y=0, line_color=C4, line_dash="dash", opacity=0.5)
                chart_layout(fig_tl, "Sentiment Timeline (per sentence)")
                fig_tl.update_layout(
                    height=320,
                    xaxis_title="Sentence Index", yaxis_title="Compound Score",
                    xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR),
                )
                st.plotly_chart(fig_tl, use_container_width=True)

        with v_tab3:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                if len(sentences) >= 3:
                    x_vals = [len(s.split()) for s in sentences]
                    y_vals = [
                        sum(max(1, len(re.findall(r"[aeiou]", w)))
                            for w in s.split()) / max(len(s.split()), 1)
                        for s in sentences
                    ]
                    fig = go.Figure(go.Scatter(
                        x=x_vals, y=y_vals, mode="markers",
                        marker=dict(
                            size=10, color=list(range(len(x_vals))),
                            colorscale=[[0, C6], [0.5, C1], [1, C4]], showscale=False,
                        ),
                    ))
                    chart_layout(fig, "Readability vs Sentence Length")
                    fig.update_xaxes(title_text="Sentence Length (words)")
                    fig.update_yaxes(title_text="Avg Syllables/Word")
                    st.plotly_chart(fig, use_container_width=True)
            with col_a2:
                word_lens = [len(w) for w in words if w.isalpha()]
                if word_lens:
                    fig = go.Figure(go.Box(
                        y=word_lens, name="Word Length", marker_color=C1,
                        line_color=C4, fillcolor=rgba_c1(0.13),
                    ))
                    chart_layout(fig, "Word Length Distribution (Box Plot)")
                    st.plotly_chart(fig, use_container_width=True)
            col_a3, col_a4 = st.columns(2)
            with col_a3:
                if pos_data:
                    pos_counts = Counter(d["pos"] for d in pos_data)
                    fig = go.Figure(go.Pie(
                        labels=list(pos_counts.keys()),
                        values=list(pos_counts.values()),
                        hole=0, marker=dict(colors=CHART_COLORS),
                    ))
                    chart_layout(fig, "POS Tag Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            with col_a4:
                if entities:
                    all_ents = [
                        (label, ent)
                        for label, ents in entities.items()
                        for ent in list(set(ents))[:3]
                    ]
                    labels  = ["Entities"] + [e[0] for e in all_ents] + [e[1] for e in all_ents]
                    parents = [""] + ["Entities"] * len(all_ents) + [e[0] for e in all_ents]
                    fig = go.Figure(go.Sunburst(
                        labels=labels, parents=parents,
                        marker=dict(colorscale=[[0, C6], [0.5, C1], [1, C4]]),
                    ))
                    chart_layout(fig, "Entity Sunburst")
                    st.plotly_chart(fig, use_container_width=True)

        with v_tab4:
            if show_network and len(tokens) >= 5:
                with st.spinner("Building co-occurrence network…"):
                    G = cooccurrence_network(tokens, top_n=40, window=3)
                    fig_net = render_network_plotly(G)
                if fig_net:
                    st.plotly_chart(fig_net, use_container_width=True)
                else:
                    st.info("Not enough edges for network visualization.")
            else:
                bigrams_list = get_bigrams(tokens[:100])
                if bigrams_list:
                    bi_freq = Counter(bigrams_list).most_common(15)
                    edge_x, edge_y = [], []
                    node_x, node_y, node_text = [], [], []
                    unique_nodes = list({n for pair, _ in bi_freq for n in pair})
                    pos_dict = {
                        n: (
                            np.cos(2 * np.pi * i / max(len(unique_nodes), 1)),
                            np.sin(2 * np.pi * i / max(len(unique_nodes), 1)),
                        )
                        for i, n in enumerate(unique_nodes)
                    }
                    for (a, b), _ in bi_freq:
                        x1, y1 = pos_dict.get(a, (0, 0))
                        x2, y2 = pos_dict.get(b, (0, 0))
                        edge_x += [x1, x2, None]; edge_y += [y1, y2, None]
                    for n, (x, y) in pos_dict.items():
                        node_x.append(x); node_y.append(y); node_text.append(n)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=edge_x, y=edge_y, mode="lines",
                        line=dict(width=1, color=rgba_c6(0.53)), hoverinfo="none",
                    ))
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y, mode="markers+text",
                        text=node_text, textposition="top center",
                        marker=dict(size=14, color=C1,
                                    line=dict(width=2, color=C4)),
                    ))
                    chart_layout(fig, "Bigram Network Graph")
                    fig.update_layout(showlegend=False, height=450)
                    st.plotly_chart(fig, use_container_width=True)

        with v_tab5:
            if show_wc and tokens:
                wc_buf = wordcloud_img(tokens)
                if wc_buf:
                    st.image(wc_buf, use_container_width=True)
            else:
                st.info("Enable Word Cloud in the sidebar or enter more text.")

        with v_tab6:
            if st.button("🤖 Fetch AI Topics & Keywords via Groq"):
                with st.spinner("Calling Groq AI…"):
                    kw_raw = run_ai_task(text_v, "keywords")
                    try:
                        kw_raw_clean = re.sub(r"```json|```", "", kw_raw).strip()
                        kw_data = json.loads(kw_raw_clean)
                        kws = kw_data.get("keywords", [])
                        badges_kw = "".join(
                            f"<span class='badge badge-blue'>{k}</span>" for k in kws
                        )
                        st.markdown(
                            f"<div class='neo-card' style='line-height:2.2;'>{badges_kw}</div>",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Keyword parse error: {e}")
                        st.markdown(
                            f"<div class='result-box'>{kw_raw}</div>",
                            unsafe_allow_html=True,
                        )

                    tp_raw = run_ai_task(text_v, "topics")
                    try:
                        tp_raw_clean = re.sub(r"```json|```", "", tp_raw).strip()
                        tp_data = json.loads(tp_raw_clean)
                        topics = tp_data.get("topics", [])
                        if topics:
                            fig_topics = go.Figure(go.Bar(
                                x=[t.get("score", 0) for t in topics],
                                y=[t.get("name", "") for t in topics],
                                orientation="h",
                                marker=dict(
                                    color=[t.get("score", 0) for t in topics],
                                    colorscale=[[0, C6], [0.5, C1], [1, C4]],
                                ),
                                text=[f"{t.get('score', 0):.2f}" for t in topics],
                                textposition="outside",
                            ))
                            chart_layout(fig_topics, "AI Topic Modeling")
                            fig_topics.update_layout(
                                height=300,
                                yaxis=dict(autorange="reversed", gridcolor=GRID_COLOR),
                            )
                            st.plotly_chart(fig_topics, use_container_width=True)
                    except Exception as e:
                        st.error(f"Topics parse error: {e}")
                        st.markdown(
                            f"<div class='result-box'>{tp_raw}</div>",
                            unsafe_allow_html=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 8 — AI Chatbot (Groq-powered)
# ═══════════════════════════════════════════════════════════════════════════════
elif "🤖 AI Chatbot" in module:
    st.markdown("<div class='section-header'>🤖 AI CHATBOT — GROQ LLaMA-3.3 70B</div>",
                unsafe_allow_html=True)
    st.markdown(f"""<div class='neo-card'>
    <p style='color:#CFD6C4;font-size:0.88rem'>
    💬 This chatbot uses <strong>Groq LLaMA-3.3-70B-Versatile</strong> for
    intelligent NLP-focused responses. Configure your <code>GROQ_API_KEY</code>
    in Streamlit secrets to enable AI features.
    </p></div>""", unsafe_allow_html=True)

    for msg in st.session_state.chat_messages:
        role_color = C1 if msg["role"] == "assistant" else C4
        align = "left" if msg["role"] == "assistant" else "right"
        icon  = "🤖" if msg["role"] == "assistant" else "👤"
        st.markdown(f"""
        <div style='text-align:{align};margin:8px 0'>
          <div style='display:inline-block;max-width:80%;
               background:{"#141b2d" if msg["role"]=="assistant" else "#1a1b2d"};
               border:1px solid {role_color};border-radius:14px;
               padding:10px 16px;text-align:left'>
            <span style='font-size:0.75rem;color:{role_color}'>{icon} {msg["role"].upper()}</span><br>
            <span style='color:{C2};font-size:0.88rem'>{msg["content"]}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    user_msg = st.text_input("Type your message…", key="chat_input")
    _, col_chat2 = st.columns([3, 1])
    with col_chat2:
        send = st.button("Send 💬", use_container_width=True)

    if send and user_msg:
        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        context = st.session_state.current_text[:3000] if st.session_state.current_text else ""
        full_q = (
            f"You are NexaLex AI, an advanced NLP assistant. "
            f"{f'Context text: {context[:1500]}' if context else ''}\n\n"
            f"User: {user_msg}"
        )
        with st.spinner("🤖 Thinking…"):
            bot_reply = run_ai_task(full_q, "qa")
        st.session_state.chat_messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()

    st.markdown("<div class='section-header'>⚡ SMART AUTO REPLY GENERATOR</div>",
                unsafe_allow_html=True)
    reply_input = st.text_area("Message to reply to:", height=100)
    if st.button("Generate Smart Reply") and reply_input:
        sent_label, _, _, _ = detect_sentiment(reply_input)
        if "?" in reply_input:
            reply = "Thank you for your question! I'll look into this and get back to you with a detailed response shortly."
        elif sent_label == "Negative":
            reply = "I sincerely apologize for the inconvenience. We take your concern very seriously and will work to resolve this immediately."
        elif sent_label == "Positive":
            reply = "Thank you so much for your kind words! We're thrilled to hear your positive feedback."
        else:
            reply = "Thank you for reaching out. I've received your message and will respond with the relevant information as soon as possible."
        st.markdown(f"<div class='result-box'>💬 {reply}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 9 — Audio & Speech
# ═══════════════════════════════════════════════════════════════════════════════
elif "🎙 Audio" in module:
    st.markdown("<div class='section-header'>🎙 AUDIO & SPEECH PROCESSING</div>",
                unsafe_allow_html=True)
    a_tab1, a_tab2 = st.tabs(["🎙 Speech to Text", "🔊 Text to Speech"])

    with a_tab1:
        st.markdown("""<div class='neo-card'>
        <p style='color:#CFD6C4;font-size:0.88rem'>
        🎙 Audio transcription uses OpenAI Whisper.<br>
        Install via: <code>pip install openai-whisper</code>
        </p></div>""", unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload Audio File",
                                      type=["mp3", "wav", "m4a", "ogg"])
        if audio_file:
            st.audio(audio_file)
            if st.button("🎙 Transcribe Audio"):
                with st.spinner("Transcribing…"):
                    try:
                        import whisper, tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                            tmp.write(audio_file.read())
                            tmp_path = tmp.name
                        model = whisper.load_model("base")
                        result = model.transcribe(tmp_path)
                        transcript = result["text"]
                        st.markdown(
                            f"<div class='result-box'>{transcript}</div>",
                            unsafe_allow_html=True,
                        )
                        st.session_state.raw_text = transcript
                        st.download_button("⬇ Download Transcript", transcript,
                                           "transcript.txt")
                        os.unlink(tmp_path)
                    except ImportError:
                        st.info("Install Whisper: `pip install openai-whisper`")
                    except Exception as e:
                        st.error(f"Transcription error: {e}")

    with a_tab2:
        tts_text = st.text_area("Text to Convert to Speech", height=150,
                                value=st.session_state.processed_text or "")
        col_tts1, col_tts2 = st.columns(2)
        with col_tts1:
            tts_lang = st.selectbox("Language", ["en", "ur", "ar", "fr", "es"])
        with col_tts2:
            tts_speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)
        if st.button("🔊 Generate Speech") and tts_text:
            try:
                from gtts import gTTS
                import tempfile
                tts = gTTS(text=tts_text, lang=tts_lang, slow=(tts_speed < 0.8))
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    tts.save(f.name)
                    with open(f.name, "rb") as af:
                        audio_bytes = af.read()
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button("⬇ Download Audio", audio_bytes,
                                   "speech.mp3", "audio/mp3")
                os.unlink(f.name)
            except ImportError:
                st.info("Install gTTS: `pip install gtts`")
            except Exception as e:
                st.error(f"TTS error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 10 — Batch Processing
# ═══════════════════════════════════════════════════════════════════════════════
elif "📦 Batch" in module:
    st.markdown("<div class='section-header'>📦 BATCH PROCESSING ENGINE</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='neo-card'>
    <p style='color:#CFD6C4;font-size:0.88rem'>
    Upload multiple <code>.txt</code> or <code>.csv</code> files for parallel NLP processing.
    Each file is analysed independently and results are compiled into a downloadable report.
    </p></div>""", unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload Files for Batch Processing",
        type=["txt", "csv"], accept_multiple_files=True, key="batch_upload",
    )
    opts_batch = {
        "html": True, "lower": True, "punct": True,
        "num": False, "emoji": False, "url": True, "email": True,
        "slang": False, "markdown": False,
    }

    if batch_files and st.button("⚡ Process All Files", use_container_width=True):
        results  = []
        progress = st.progress(0)
        status   = st.empty()
        for i, bf in enumerate(batch_files):
            status.markdown(
                f"<div class='loading-pulse'>Processing: {bf.name}…</div>",
                unsafe_allow_html=True,
            )
            try:
                raw = bf.read().decode()
                cleaned = clean_text(raw, opts_batch)
                tokens, _, entities = process_text_nlp(cleaned, True, True)
                kw = extract_keywords(cleaned, 5)
                sent_label, sent_score, _, _ = detect_sentiment(cleaned)
                tox, _ = detect_toxic(cleaned)
                rs, rg = readability_score(cleaned)
                results.append({
                    "File":            bf.name,
                    "Original Words":  len(raw.split()),
                    "Processed Words": len(tokens),
                    "Reduction %":     round((1 - len(tokens) / max(len(raw.split()), 1)) * 100, 1),
                    "Sentiment":       sent_label,
                    "Sentiment Score": sent_score,
                    "Toxicity Score":  tox,
                    "Readability Score": rs,
                    "Readability Grade": rg,
                    "Top Keywords":    ", ".join(k[0] for k in kw[:5]),
                    "Entity Count":    sum(len(v) for v in entities.values()),
                })
            except Exception as e:
                results.append({"File": bf.name, "Error": str(e)})
            progress.progress((i + 1) / len(batch_files))
            time.sleep(0.1)

        status.empty()
        st.success(f"✅ Processed {len(batch_files)} files!")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        if len(results) >= 2 and "Sentiment Score" in df_results.columns:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig = go.Figure(go.Bar(
                    x=df_results["File"], y=df_results["Sentiment Score"],
                    marker=dict(
                        color=df_results["Sentiment Score"],
                        colorscale=[[0, C4], [0.5, C5], [1, C1]],
                    ),
                ))
                chart_layout(fig, "Sentiment Scores by File")
                st.plotly_chart(fig, use_container_width=True)
            with col_b2:
                fig = go.Figure(go.Scatter(
                    x=df_results["Original Words"], y=df_results["Readability Score"],
                    mode="markers+text", text=df_results["File"],
                    marker=dict(size=14, color=CHART_COLORS[: len(df_results)]),
                ))
                chart_layout(fig, "Readability vs Length")
                st.plotly_chart(fig, use_container_width=True)

        col_dl_b1, col_dl_b2 = st.columns(2)
        with col_dl_b1:
            st.download_button(
                "⬇ Download Report (.csv)",
                df_results.to_csv(index=False),
                "batch_report.csv",
                use_container_width=True,
            )
        with col_dl_b2:
            excel_buf = io.BytesIO()
            df_results.to_excel(excel_buf, index=False, engine="openpyxl")
            st.download_button(
                "⬇ Download Report (.xlsx)",
                excel_buf.getvalue(),
                "batch_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:30px 0 10px;
            border-top:1px solid {LINE_COLOR};margin-top:40px'>
  <span style='font-family:Orbitron;font-size:0.75rem;
               color:{C6};letter-spacing:3px'>
    NEXALEX AI · ADVANCED NLP STUDIO · GROQ LLaMA-3.3 · SPACY · STREAMLIT
  </span>
</div>
""", unsafe_allow_html=True)
