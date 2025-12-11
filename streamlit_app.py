# ================================================
#  📱 PhishX — English + Hinglish SMS Phishing Detector
# ================================================

import os
import joblib
import re
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)

# ------------------------------
#  STYLE
# ------------------------------
st.set_page_config(page_title="PhishX", page_icon="📱", layout="centered")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #0f172a, #1e293b);
    color: #f8fafc;
}
h1, h2, h3 {color: #f8fafc !important;}
textarea, .stButton>button {
    border-radius: 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
#  LANGUAGE DETECTOR (Hugging Face)
# ------------------------------
@st.cache_resource
def load_lang_model():
    model_name = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

lang_tokenizer, lang_model = load_lang_model()

def detect_language(text: str):
    text = text.strip()
    if not text:
        return "unknown"

    # Pre-clean to remove noise
    clean = re.sub(r"http\S+|\$|\d+|[^A-Za-z\s]", "", text)
    if not clean:
        return "unknown"

    inputs = lang_tokenizer(clean, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = lang_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = probs.argmax(dim=1).item()
        confidence = probs[0, pred_id].item()
        lang_label = lang_model.config.id2label[pred_id].lower()

    if "hindi" in lang_label or "roman" in lang_label:
        lang = "hinglish"
    elif "english" in lang_label:
        lang = "english"
    else:
        lang = "unknown"

    # Force English if low confidence and message looks short / spammy
    if confidence < 0.6 and lang == "unknown":
        lang = "english"

    return lang
# ------------------------------
#  LOAD MODELS
# ------------------------------
@st.cache_resource
def load_models():
    base = "models"
    nb_model = joblib.load(os.path.join(base, "spam_nb_model.joblib"))
    vectorizer = joblib.load(os.path.join(base, "tfidf_vectorizer.joblib"))
    threshold = joblib.load(os.path.join(base, "spam_threshold.joblib"))

    tokenizer, bert_model = None, None
    try:
        tokenizer = BertTokenizer.from_pretrained(base)
        bert_model = BertForSequenceClassification.from_pretrained(base)
        bert_model.eval()
    except Exception as e:
        st.warning(f"⚠️ BERT model not loaded ({e}). Using only Naive Bayes.")

    return nb_model, vectorizer, threshold, tokenizer, bert_model

nb_model, vectorizer, threshold, tokenizer, bert_model = load_models()

# ------------------------------
#  PREDICTION
# ------------------------------
def predict_message(msg):
    lang = detect_language(msg)

    if lang == "english" and bert_model is not None:
        inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            proba = torch.softmax(outputs.logits, dim=1)[0, 1].item()
    else:
        X = vectorizer.transform([msg])
        proba = nb_model.predict_proba(X)[0, 1]

    label = "Spam" if proba >= threshold else "Ham"
    return lang, label, proba

# ------------------------------
#  CUSTOM CONFIDENCE BAR
# ------------------------------
def confidence_bar(prob):
    if prob < 0.4:
        color = "#4CAF50"
        emoji = "✅"
        label = "Ham"
    elif prob < 0.75:
        color = "#FFC107"
        emoji = "⚠️"
        label = "Suspicious"
    else:
        color = "#F44336"
        emoji = "🚨"
        label = "Spam"

    bar_html = f"""
    <div style="border-radius: 8px; background-color: #ddd; height: 25px; width: 100%; position: relative;">
        <div style="
            background-color: {color};
            width: {prob * 100}%;
            height: 100%;
            border-radius: 8px;
            transition: width 0.3s ease;">
        </div>
        <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%);
                    color: black; font-weight: 600;">
            {emoji} {label} ({prob:.2f})
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

# ------------------------------
#  ADVANCED MODE HELPERS
# ------------------------------
def show_tfidf_explanation(text):
    X = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # For MultinomialNB, feature importance is in log probabilities
    spam_log_prob = nb_model.feature_log_prob_[1]
    ham_log_prob = nb_model.feature_log_prob_[0]
    importance = spam_log_prob - ham_log_prob  # relative contribution to spam

    top_indices = np.argsort(importance)[-20:]
    important_words = {feature_names[i]: float(importance[i]) for i in top_indices}

    st.subheader("🔍 Top TF-IDF Features Influencing Spam Prediction")
    wc = WordCloud(width=700, height=300, background_color="black",
                   colormap="Reds").generate_from_frequencies(important_words)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


def show_bert_attention_placeholder():
    st.subheader("🧠 BERT Attention Visualization")
    st.caption("Attention heatmaps can be generated using the model’s attention tensors. "
               "To keep deployment lightweight, this demo uses a placeholder.")
    st.image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/bert_architecture.png",
        use_container_width=True
    )

# ------------------------------
#  STREAMLIT UI
# ------------------------------
tab_main, tab_adv = st.tabs(["🔎 Detect Phishing SMS", "🧩 Advanced Mode"])

# === MAIN TAB ===
with tab_main:
    st.title("📱 PhishX — SMS Phishing Detector")
    st.write("Detect English & Hinglish phishing or spam messages using AI models.")

    sms = st.text_area("✉️ Enter an SMS message:", height=120)

    if st.button("Analyze"):
        if not sms.strip():
            st.warning("Please enter a message first.")
        else:
            lang, label, proba = predict_message(sms)
            st.write(f"🌐 Detected Language: **{lang.upper()}**")
            confidence_bar(proba)

# === ADVANCED TAB ===
with tab_adv:
    st.title("🧩 Advanced Model Insights")
    st.write("Visualize model reasoning and feature importance.")

    msg = st.text_area("Enter a message to inspect:", height=120, key="adv_msg")

    if st.button("Explain"):
        if not msg.strip():
            st.warning("Enter text to explain.")
        else:
            lang = detect_language(msg)
            st.write(f"🌐 Language: **{lang.upper()}**")

            # Let user pick visualization
            mode = st.radio("Select explanation type:", ["TF-IDF (Naive Bayes)", "BERT Attention"])
            if mode == "TF-IDF (Naive Bayes)":
                show_tfidf_explanation(msg)
            else:
                show_bert_attention_placeholder()

st.markdown("---")
st.caption("Built with ❤️ using XLM-R (Language Detection) + BERT + Naive Bayes (TFIDF).")
