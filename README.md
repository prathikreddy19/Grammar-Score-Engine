# Grammar Scoring Engine (SHL Assessment)

## Overview
This project was developed as part of the SHL Research Internship Assessment:  
**Build a Grammar Scoring Engine from Voice Samples.**

The system evaluates the grammatical proficiency of spoken English from audio recordings and assigns a score between **1 and 5**.  
It uses acoustic and semantic audio features during training (Whisper + Librosa) and a lightweight Librosa-only model for real-time deployment.

Live Demo:  
🔗 **[Grammar Scoring Engine Web App](https://prathikreddy19-grammar-score-engine.streamlit.app/)**

---

## Key Highlights
- Predicts grammar proficiency from `.wav` and `.mp3` files.  
- Trained hybrid model using **Whisper**, **SentenceTransformer**, and **Librosa**.  
- Deployed lightweight model (Librosa-only) for fast inference.  
- Model trained using **XGBoost** with precomputed hybrid embeddings.  
- Elegant, interactive UI built with **Streamlit**.

## Technical Approach

### 1. Data and Feature Extraction
- Dataset: 444 labeled audio samples from SHL Grammar dataset.  
- Extracted acoustic features using **Librosa** (MFCCs, spectral centroid, bandwidth, rolloff, RMS, ZCR).  
- Generated transcriptions with **OpenAI Whisper** and encoded semantics using **SentenceTransformer**.  

### 2. Model Training
- Combined all feature vectors into a hybrid dataset.
- Model: **XGBoost Regressor**
- Performance:  
  - Mean Absolute Error (MAE): **0.63**  
  - R² Score: **0.55**  

### 3. Deployment Optimization
- Whisper and text embeddings were excluded for runtime efficiency.  
- The Streamlit app loads **precomputed Librosa embeddings** for instant scoring.  
- Model file: `xgb_hybrid_model.pkl`  
- Precomputed arrays: `utils/X_hybrid.npy`, `utils/y_hybrid.npy`, `utils/filenames.npy`.
  
## 4. Project Structure
SHL-Grammar-Scoring-Engine/
│
├── app.py # Streamlit UI for scoring
├── xgb_hybrid_model.pkl # Trained XGBoost model
├── utils/
│ ├── X_hybrid.npy # Precomputed features
│ ├── y_hybrid.npy # Corresponding labels
│ └── filenames.npy # Audio filenames
├── Grammar_score_training.ipynb # Training notebook (Whisper + Librosa)
├── requirements.txt # Dependencies
└── README.md

Deployment

Platform: Streamlit Cloud
Live App: https://prathikreddy19-grammar-score-engine.streamlit.app/
Model Hosting: Bundled directly with Streamlit app
GitHub Repository: Publicly accessible for review

## Running Locally

### 1. Clone Repository

**2. Install Dependencies**
pip install -r requirements.txt

**3. Launch the App**
streamlit run app.py

**4. Test**

Upload a .wav or .mp3 file in the web interface to receive a grammar score (1–5) with proficiency percentage.

**Requirements**
streamlit
numpy
joblib
librosa
xgboost
soundfile

**Notes**

The deployed app uses Librosa-only inference for instant results.
Whisper and SentenceTransformer-based features were used in the training notebook only.
All artifacts are included for reproducibility and offline testing.

## Author
**Prathik Reddy**  
Email: [prathikreddy1230@gmail.com](mailto:prathikreddy1230@gmail.com)  
LinkedIn: [linkedin.com/in/prathikreddymettu](https://linkedin.com/in/prathikreddymettu)

