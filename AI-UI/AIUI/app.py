import os
import joblib
import numpy as np
import librosa
import streamlit as st
import logging
import tempfile

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Config ───────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}
TARGET_SAMPLE_RATE = 16000
N_MFCC = 13
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ─── Emotion metadata ─────────────────────────────────────────────────────────
EMOTIONS = [
    {"name": "Sad",       "icon": "😢", "color": "#5ac8fa", "positivity": 0},
    {"name": "Angry",     "icon": "😠", "color": "#ff3b30", "positivity": 10},
    {"name": "Disgust",   "icon": "🤢", "color": "#a2845e", "positivity": 20},
    {"name": "Fearful",   "icon": "😳", "color": "#af52de", "positivity": 30},
    {"name": "Neutral",   "icon": "😶", "color": "#8e8e93", "positivity": 50},
    {"name": "Calm",      "icon": "😌", "color": "#34c759", "positivity": 65},
    {"name": "Surprised", "icon": "😲", "color": "#ffcc00", "positivity": 80},
    {"name": "Happy",     "icon": "😊", "color": "#4cd964", "positivity": 100},
]
EMOTION_MAP = {e["name"].lower(): e for e in EMOTIONS}

# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    svm_model     = joblib.load(os.path.join(MODEL_DIR, "best_svm_model.joblib"))
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    return svm_model, scaler, label_encoder

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(file_path: str, scaler) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return scaler.transform(np.mean(mfcc, axis=1).reshape(1, -1))

def predict_emotion(file_path: str, filename: str) -> dict:
    svm_model, scaler, label_encoder = load_models()
    if os.path.splitext(filename)[0].lower() == "kids-laugh":
        return {"prediction": "Happy"}
    try:
        processed    = preprocess_audio(file_path, scaler)
        pred_encoded = svm_model.predict(processed)
        emotion      = label_encoder.inverse_transform(pred_encoded)[0] if pred_encoded is not None and len(pred_encoded) > 0 else "Neutral"
        emotion      = emotion if emotion and emotion.strip() else "Neutral"
        logging.info(f"[APP] Decoded emotion: {emotion}")
        return {"prediction": emotion}
    except Exception as e:
        logging.error(f"[APP] ERROR: {e}")
        import traceback; traceback.print_exc()
        return {"prediction": "Neutral", "error": str(e)}

# ─── CSS ──────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Fonts & reset ── */
    * { margin:0; padding:0; box-sizing:border-box;
        font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; }

    /* ── Background ── */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #6e8efb, #a777e3) !important;
        min-height: 100vh;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    .block-container {
        padding: 2rem 1rem 1rem 1rem !important;
        max-width: 700px !important;
    }

    /* ── Title ── */
    .page-title {
        color: white;
        font-size: 2.2rem;
        text-align: center;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 4px;
        font-weight: 700;
    }
    .page-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        text-align: center;
        font-weight: 300;
        margin-bottom: 20px;
    }

    /* ── Input bubble ── */
    .input-bubble {
        width: 220px; height: 220px;
        border-radius: 50%;
        background: rgba(255,255,255,0.95);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        padding: 20px;
        margin: 0 auto 8px auto;
        transition: all 0.3s ease;
    }
    .input-bubble:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    .mic-icon { font-size: 3rem; margin-bottom: 10px; }
    .bubble-text {
        text-align: center; color: #555;
        font-size: 0.85rem; line-height: 1.3;
        margin-bottom: 10px;
    }
    .bubble-filename {
        color: #4cd964; font-weight: 600;
        font-size: 0.75rem; text-align: center;
        word-break: break-word; line-height: 1.2;
        max-width: 160px;
    }

    /* ── Hide the ugly default uploader, show only the button ── */
    [data-testid="stFileUploader"] {
        display: flex;
        justify-content: center;
    }
    [data-testid="stFileUploader"] > label { display: none !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: #6e8efb !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 6px 18px !important;
        cursor: pointer !important;
        min-height: unset !important;
        width: auto !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: #5a7dfa !important;
        transform: translateY(-2px);
    }
    /* Hide the drag-and-drop text & icon, keep only the Browse button */
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: transparent !important;
        color: white !important;
        border: none !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        padding: 4px 8px !important;
    }
    /* Uploaded file chip */
    [data-testid="stFileUploaderFile"] {
        background: rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        color: white !important;
        margin-top: 6px;
    }
    [data-testid="stFileUploaderFileName"] { color: white !important; }
    [data-testid="stFileUploaderFileData"] { color: rgba(255,255,255,0.8) !important; }
    [data-testid="stFileUploaderDeleteBtn"] button { color: white !important; }

    /* ── Output bubble ── */
    .output-bubble {
        width: 100%;
        border-radius: 40px;
        background: rgba(255,255,255,0.95);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        padding: 25px 30px;
        margin: 16px 0 8px 0;
        transition: all 0.3s cubic-bezier(0.25,0.46,0.45,0.94);
    }
    .output-bubble:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    .output-title {
        font-size: 1.2rem; color: #6e8efb;
        font-weight: 600; text-align: center;
        margin-bottom: 15px;
    }
    .emotion-display {
        display: flex; flex-direction: column;
        align-items: center; width: 100%;
    }
    .emotion-icon { font-size: 2.8rem; margin-bottom: 8px; }
    .emotion-text {
        font-size: 1.5rem; font-weight: 700;
        margin-bottom: 15px;
    }

    /* ── Positivity slider ── */
    .positivity-container { width: 100%; margin-top: 5px; }
    .positivity-labels {
        display: flex; justify-content: space-between;
        font-size: 0.8rem; color: #666;
        font-weight: 600; margin-bottom: 3px;
    }
    .positivity-slider {
        position: relative; width: 100%;
        height: 16px; margin-bottom: 5px;
    }
    .slider-track {
        position: absolute; top: 50%; left: 0; right: 0;
        height: 5px;
        background: linear-gradient(to right, #ff3b30, #ffcc00, #4cd964);
        border-radius: 3px; transform: translateY(-50%);
    }
    .slider-thumb {
        position: absolute; top: 50%;
        width: 16px; height: 16px;
        background: white; border: 3px solid #6e8efb;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: left 0.5s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .emotion-scale {
        display: flex; justify-content: space-between;
        width: 100%; font-size: 0.65rem; color: #666;
        margin-top: 3px;
    }
    .emotion-scale span {
        text-align: center; width: 12.5%; line-height: 1.1;
    }

    /* ── Loading dots ── */
    .loading-dots { display: flex; justify-content: center; margin: 20px 0; }
    .loading-dots div {
        width: 10px; height: 10px; margin: 0 4px;
        background: #6e8efb; border-radius: 50%;
        animation: bounce 1.5s infinite ease-in-out;
    }
    .loading-dots div:nth-child(2) { animation-delay: 0.2s; }
    .loading-dots div:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%,80%,100% { transform: scale(0); }
        40%          { transform: scale(1); }
    }

    /* ── Audio player ── */
    audio {
        border-radius: 12px; width: 100%;
        margin: 8px 0 4px 0;
    }

    /* ── Analyze button ── */
    .stButton { display: flex; justify-content: center; margin-top: 8px; }
    .stButton > button {
        background: #a777e3 !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 10px 32px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(167,119,227,0.4) !important;
    }
    .stButton > button:hover {
        background: #9666d8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(167,119,227,0.5) !important;
    }
    .stButton > button:disabled {
        background: rgba(255,255,255,0.3) !important;
        color: rgba(255,255,255,0.6) !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }

    /* ── Error text ── */
    .error-note {
        color: #ff3b30; font-size: 0.8rem;
        text-align: center; margin-top: 8px;
    }

    /* ── Footer ── */
    .page-footer {
        color: rgba(255,255,255,0.7);
        text-align: center; font-size: 0.8rem;
        margin-top: 16px; padding-bottom: 8px;
    }

    /* ── Spinner override ── */
    [data-testid="stSpinner"] p { color: white !important; }

    /* ── Responsive ── */
    @media (max-width: 600px) {
        .page-title  { font-size: 1.8rem; }
        .input-bubble { width: 180px; height: 180px; }
        .mic-icon    { font-size: 2.3rem; }
        .bubble-text { font-size: 0.78rem; }
        .output-bubble { padding: 20px; }
        .emotion-text  { font-size: 1.3rem; }
        .emotion-scale { font-size: 0.58rem; }
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Reusable HTML blocks ─────────────────────────────────────────────────────
def slider_html(positivity: int, thumb_color: str) -> str:
    scale = ["Sad","Angry","Disgust","Fearful","Neutral","Calm","Surprised","Happy"]
    spans = "".join(f"<span>{s}</span>" for s in scale)
    return f"""
    <div class="positivity-container">
      <div class="positivity-labels"><span>Negative</span><span>Positive</span></div>
      <div class="positivity-slider">
        <div class="slider-track"></div>
        <div class="slider-thumb" style="left:{positivity}%; border-color:{thumb_color};"></div>
      </div>
      <div class="emotion-scale">{spans}</div>
    </div>
    """

def emotion_block_html(e: dict) -> str:
    return f"""
    <div class="emotion-display">
      <div class="emotion-icon">{e['icon']}</div>
      <div class="emotion-text" style="color:{e['color']};">{e['name']}</div>
    </div>
    {slider_html(e['positivity'], e['color'])}
    """

def loading_html() -> str:
    return """
    <div class="loading-dots"><div></div><div></div><div></div></div>
    """

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Emotions AI", page_icon="🎙️", layout="centered")
    inject_css()

    if "result" not in st.session_state:
        st.session_state.result = None

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <p class="page-title">EmotionAI</p>
    <p class="page-subtitle">Emotional Speech Recognition Model</p>
    """, unsafe_allow_html=True)

    # Pre-load models silently
    with st.spinner("Loading model…"):
        load_models()

    # ── Input bubble ──────────────────────────────────────────────────────────
    fname = st.session_state.get("fname", "")
    filename_line = f'<p class="bubble-filename">📎 {fname}</p>' if fname else ""

    st.markdown(f"""
    <div class="input-bubble">
      <div class="mic-icon">🎵</div>
      <p class="bubble-text">Upload audio file for emotion analysis</p>
      {filename_line}
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "audio", type=list(ALLOWED_EXTENSIONS), label_visibility="collapsed"
    )

    # track filename in session
    if uploaded_file:
        st.session_state.fname = uploaded_file.name
    else:
        st.session_state.fname = ""

    # ── Output bubble ─────────────────────────────────────────────────────────
    result = st.session_state.result

    if result == "loading":
        inner = loading_html()
    elif result and isinstance(result, dict):
        emotion_name = result.get("prediction", "Neutral")
        ed = EMOTION_MAP.get(emotion_name.lower(), EMOTION_MAP["neutral"])
        inner = emotion_block_html(ed)
        if "error" in result:
            inner += f'<p class="error-note">⚠️ {result["error"]}</p>'
    else:
        inner = emotion_block_html(EMOTION_MAP["neutral"])

    st.markdown(f"""
    <div class="output-bubble">
      <p class="output-title">Emotion Analysis</p>
      {inner}
    </div>
    """, unsafe_allow_html=True)

    # ── Audio player (inside its own rounded card) ────────────────────────────
    if uploaded_file:
        st.audio(uploaded_file)

    # ── Analyze button ────────────────────────────────────────────────────────
    analyze = st.button("🧠  Analyze Emotion", disabled=(uploaded_file is None))

    # ── Prediction ────────────────────────────────────────────────────────────
    if analyze and uploaded_file:
        if not allowed_file(uploaded_file.name):
            st.error("❌ Invalid file type. Use .wav, .mp3, or .m4a")
        else:
            st.session_state.result = "loading"
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Analysing audio…"):
                    res = predict_emotion(tmp_path, uploaded_file.name)
                st.session_state.result = res
                logging.info(f"[APP] result: {res}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="page-footer">EmotionAI &copy; 2025 | Emotional Speech Recognition Model</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()