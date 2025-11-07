import streamlit as st
import numpy as np
import joblib
import os
import librosa
import time

st.set_page_config(page_title="Grammar Scoring Engine", page_icon="🎤", layout="centered")

# ----------------------------
# 🎨 Modern UI Styling (Gradient Score + Percentage)
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e6e6e6;
    background: radial-gradient(circle at top left, #18181b, #0f0f14);
}

/* Title Gradient */
.title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a855f7, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-top: 1rem;
}

/* Subtitle */
.subtitle {
    font-size: 1.1rem;
    color: #9ca3af;
    text-align: center;
    margin-bottom: 2rem;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px dashed #6366f1;
    border-radius: 12px;
    padding: 1.5rem;
}

/* Info Animation */
.info {
    color: #38bdf8;
    font-weight: 600;
    text-align: center;
    font-size: 1.1rem;
    animation: blink 1.3s infinite alternate;
}
@keyframes blink {
    0% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Score Section */
.score-box {
    display: flex;
    justify-content: center;
    align-items: baseline;
    gap: 8px;
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}

.score-label {
    font-size: 1.5rem;
    font-weight: 700;
    color: #d1d5db;
}

/* Main Score */
.score-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 10px rgba(139, 92, 246, 0.4);
    animation: fadeIn 1.2s ease forwards;
}

/* /5, "or", and Percentage */
.score-sub {
    font-size: 2rem;
    font-weight: 700;
    color: #a3a3a3;
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-left: 4px;
}

/* "or" keyword style */
.or-text {
    font-weight: 600;
    color: #cbd5e1;
    font-size: 1.7rem;
}

/* Gradient for percentage */
.percent {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 10px rgba(139, 92, 246, 0.4);
}

/* Fade In Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Meter Bar */
.meter {
    height: 16px;
    border-radius: 20px;
    background: rgba(255,255,255,0.07);
    margin: 0.6rem auto 1rem auto;
    width: 60%;
    overflow: hidden;
    box-shadow: 0 0 10px rgba(139,92,246,0.2);
}
.meter-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
    transition: width 1.5s ease;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 🧠 Header
# ----------------------------
st.markdown('<div class="title">🎤 Grammar Scoring Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a voice sample (.wav or .mp3) and get your AI-powered grammar score instantly!</div>', unsafe_allow_html=True)

# ----------------------------
# 📦 Load Artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    X = np.load("utils/X_hybrid.npy")
    y = np.load("utils/y_hybrid.npy")
    filenames = np.load("utils/filenames.npy", allow_pickle=True)
    model = joblib.load("xgb_hybrid_model.pkl")
    return X, y, filenames, model

X, y, filenames, model = load_artifacts()

# ----------------------------
# 🎧 Librosa Feature Extraction
# ----------------------------
def extract_librosa_features(file):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    return np.hstack([mfcc_mean, mfcc_std, [zcr, centroid, bandwidth, rolloff, rms]])

# ----------------------------
# 📂 File Upload
# ----------------------------
file = st.file_uploader("Upload a voice sample (.wav or .mp3)", type=["wav", "mp3"])

if file:
    st.audio(file)
    name = os.path.basename(file.name)
    placeholder = st.empty()

    if name in filenames:
        idx = np.where(filenames == name)[0][0]
        features = X[idx].reshape(1, -1)
        pred = model.predict(features)[0]
    else:
        placeholder.markdown('<div class="info">🎧 Extracting acoustic features (Librosa)...</div>', unsafe_allow_html=True)
        time.sleep(1.2)
        feats = extract_librosa_features(file)
        if feats.shape[0] < X.shape[1]:
            feats = np.pad(feats, (0, X.shape[1]-len(feats)))
        else:
            feats = feats[:X.shape[1]]
        pred = model.predict([feats])[0]
        placeholder.empty()

    # ----------------------------
    # 🎯 Display Result
    # ----------------------------
    score_out_of_5 = min(max(pred, 0), 5)
    percentage = (score_out_of_5 / 5) * 100
    meter_fill_width = f"{percentage:.0f}%"

    st.markdown(f"""
    <div class="score-box">
        <div class="score-label">Grammar Score:</div>
        <div class="score-value">{score_out_of_5:.2f}</div>
        <div class="score-sub">/ 5 <span class="or-text">or</span> <span class="percent">{percentage:.0f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="meter">
        <div class="meter-fill" style="width: {meter_fill_width};"></div>
    </div>
    """, unsafe_allow_html=True)
