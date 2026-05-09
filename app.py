# ==========================================
# APP.PY — STREAMLIT DEPLOYMENT APP
# ==========================================
# This file is deployed to Render.
# It ONLY loads pre-trained models and performs prediction.
# NO model training happens here.
#
# Pre-requisites (run train.py locally first):
#   models/  ← .pkl files + ann_model.keras
#   outputs/ ← PNG plots + model_results.csv
# ==========================================

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ASD Screening — Children",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# PATHS
# ==========================================
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"

# Must match filenames used in train.py
MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree":       "Decision_Tree.pkl",
    "Random Forest":       "Random_Forest.pkl",
    "KNN":                 "KNN.pkl",
    "SVM (Linear)":        "SVM_Linear.pkl",
    "SVM (RBF)":           "SVM_RBF.pkl",
    "Naive Bayes":         "Naive_Bayes.pkl",
    "LDA":                 "LDA.pkl",
}

# ==========================================
# CACHED MODEL LOADING
# (runs once, cached for all subsequent users)
# ==========================================
@st.cache_resource(show_spinner="Loading models…")
def load_all_models():
    ml_models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            ml_models[name] = joblib.load(path)

    ann_path = os.path.join(MODELS_DIR, "ann_model.h5")
    if os.path.exists(ann_path):
        ml_models["ANN"] = load_model(ann_path, compile=False)

    return ml_models


@st.cache_resource(show_spinner="Loading preprocessing objects…")
def load_preprocessing():
    scaler          = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoders  = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    feature_names   = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    return scaler, label_encoders, feature_names, best_model_name


# ==========================================
# HELPER — build scaled input array
# ==========================================
def build_input(a_scores, age, sex, jaundice, family_asd,
                feature_names, label_encoders, scaler):
    """
    Convert raw form values into a scaled numpy array
    ready for model prediction.
    """
    raw = {
        "A1": a_scores[0],  "A2": a_scores[1],  "A3": a_scores[2],
        "A4": a_scores[3],  "A5": a_scores[4],  "A6": a_scores[5],
        "A7": a_scores[6],  "A8": a_scores[7],  "A9": a_scores[8],
        "A10": a_scores[9],
        "Age":        age,
        "Sex":        sex,
        "Jauundice":  jaundice,   # keeps the original (misspelled) column name
        "Family_ASD": family_asd,
    }

    # Encode categorical features using saved LabelEncoders
    for col, le in label_encoders.items():
        if col in raw:
            raw[col] = int(le.transform([raw[col]])[0])

    # Arrange values in the exact column order used during training
    arr = np.array([[raw[f] for f in feature_names]])
    return scaler.transform(arr)


# ==========================================
# LOAD EVERYTHING
# ==========================================
all_models = load_all_models()
scaler, label_encoders, feature_names, best_model_name = load_preprocessing()

# ==========================================
# HEADER
# ==========================================
st.title("🧠 ASD Screening in Children")
st.markdown(
    "**Autism Spectrum Disorder prediction** using 9 pre-trained Machine Learning models.  "
    "_This tool is for screening purposes only — not a clinical diagnosis._"
)
st.markdown("---")

loaded_count = len(all_models)
st.caption(f"✅ {loaded_count} models loaded from `models/`  |  Best performer: **{best_model_name}**")

# ==========================================
# TABS
# ==========================================
tab_predict, tab_performance, tab_about = st.tabs(
    ["🔍 Predict", "📊 Model Performance", "ℹ️ About"]
)

# ──────────────────────────────────────────
# TAB 1 — PREDICTION
# ──────────────────────────────────────────
with tab_predict:
    st.header("Patient Input")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Behavioural Screening Questions (AQ-10)")
        st.caption("Select **1 = Yes** or **0 = No** for each question")

        aq_questions = [
            "A1 — Child looks at you when called by name",
            "A2 — Easy to get eye contact with child",
            "A3 — Child points to indicate interest in something",
            "A4 — Child points to show you things",
            "A5 — Child engages in pretend / imaginative play",
            "A6 — Child follows where you are looking",
            "A7 — Child shows comfort if you or someone is upset",
            "A8 — Child's first words were typical / normal",
            "A9 — Child uses simple gestures (wave, clap, etc.)",
            "A10 — Child stares at nothing with no apparent purpose",
        ]

        a_scores = []
        for q in aq_questions:
            val = st.selectbox(q, [0, 1],
                               format_func=lambda x: "0 — No" if x == 0 else "1 — Yes",
                               key=q)
            a_scores.append(val)

    with col_right:
        st.subheader("Demographic Information")

        age = st.number_input("Age (years)", min_value=1, max_value=18, value=5, step=1)

        sex = st.radio("Sex", ["m", "f"],
                       format_func=lambda x: "Male" if x == "m" else "Female",
                       horizontal=True)

        jaundice = st.radio("Born with Jaundice?", ["no", "yes"],
                            format_func=str.capitalize, horizontal=True)

        family_asd = st.radio("Family history of ASD?", ["no", "yes"],
                              format_func=str.capitalize, horizontal=True)

        st.markdown("---")
        st.subheader("Model Selection")
        model_options = list(all_models.keys())
        default_idx   = model_options.index(best_model_name) if best_model_name in model_options else 0
        chosen_model  = st.selectbox(
            "Select model for primary prediction",
            model_options,
            index=default_idx,
            help="Defaults to the best-performing model on the training dataset.",
        )

    predict_btn = st.button("🔮 Run Prediction", type="primary", use_container_width=True)

    if predict_btn:
        input_scaled = build_input(
            a_scores, age, sex, jaundice, family_asd,
            feature_names, label_encoders, scaler,
        )

        # ── Primary prediction ──────────────────────
        model_obj = all_models[chosen_model]
        if chosen_model == "ANN":
            prob = float(model_obj.predict(input_scaled, verbose=0)[0][0])
        else:
            prob = float(model_obj.predict_proba(input_scaled)[0][1])

        pred          = 1 if prob >= 0.5 else 0
        confidence_pct = prob * 100 if pred == 1 else (1 - prob) * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            if pred == 1:
                st.error("### 🔴 ASD — Positive")
                st.write("Model predicts **ASD is likely**.")
            else:
                st.success("### 🟢 No ASD — Negative")
                st.write("Model predicts **ASD is unlikely**.")
        with res_c2:
            st.metric("Confidence", f"{confidence_pct:.1f}%")
        with res_c3:
            st.metric("Model Used", chosen_model)

        # ── All-model comparison ────────────────────
        st.markdown("---")
        st.subheader("Predictions from All Models")

        all_preds = []
        for mname, mobj in all_models.items():
            try:
                if mname == "ANN":
                    p = float(mobj.predict(input_scaled, verbose=0)[0][0])
                else:
                    p = float(mobj.predict_proba(input_scaled)[0][1])
                pred_label = "ASD" if p >= 0.5 else "No ASD"
                conf_val   = p if p >= 0.5 else 1 - p
                all_preds.append({
                    "Model":      mname,
                    "Prediction": pred_label,
                    "Confidence": f"{conf_val * 100:.1f}%",
                })
            except Exception as exc:
                all_preds.append({"Model": mname, "Prediction": "Error", "Confidence": str(exc)})

        preds_df  = pd.DataFrame(all_preds)
        asd_count = sum(1 for r in all_preds if r["Prediction"] == "ASD")

        st.dataframe(preds_df, use_container_width=True, hide_index=True)
        st.info(
            f"**{asd_count} out of {len(all_preds)} models predict ASD** "
            f"({'majority' if asd_count > len(all_preds) // 2 else 'minority'})."
        )

        st.warning(
            "⚠️ This result is for **informational / screening purposes only**.  "
            "Please consult a qualified healthcare professional for a clinical diagnosis."
        )

# ──────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ──────────────────────────────────────────
with tab_performance:
    st.header("Model Performance on Test Data")
    st.caption("Metrics and plots generated during local training (train.py).")

    results_csv = os.path.join(OUTPUTS_DIR, "model_results.csv")
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)
        display_df = results_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

        # Format numeric columns as percentages for readability
        pct_cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "ROC-AUC"]
        for c in pct_cols:
            display_df[c] = display_df[c].apply(lambda v: f"{v * 100:.2f}%")
        display_df["Log Loss"] = display_df["Log Loss"].apply(lambda v: f"{float(v):.4f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("model_results.csv not found. Run train.py locally first.")

    st.markdown("---")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        roc_png = os.path.join(OUTPUTS_DIR, "roc_curves.png")
        if os.path.exists(roc_png):
            st.image(roc_png, caption="ROC Curves — All Models", use_container_width=True)

    with col_p2:
        bar_png = os.path.join(OUTPUTS_DIR, "model_comparison.png")
        if os.path.exists(bar_png):
            st.image(bar_png, caption="Model Comparison (Accuracy / F1 / ROC-AUC)",
                     use_container_width=True)

    best_cm_png = os.path.join(OUTPUTS_DIR, "confusion_matrix_best.png")
    if os.path.exists(best_cm_png):
        st.image(best_cm_png,
                 caption=f"Confusion Matrix — Best Model ({best_model_name})",
                 use_container_width=True)

    ann_hist_png = os.path.join(OUTPUTS_DIR, "ann_training_history.png")
    if os.path.exists(ann_hist_png):
        st.image(ann_hist_png, caption="ANN Training History", use_container_width=True)

    # Individual confusion matrices
    st.markdown("---")
    st.subheader("Individual Confusion Matrices")
    cm_cols = st.columns(3)
    all_cm_names = list(MODEL_FILES.keys()) + ["ANN"]
    for i, mname in enumerate(all_cm_names):
        if mname == "ANN":
            cm_file = os.path.join(OUTPUTS_DIR, "cm_ANN.png")
        else:
            safe = MODEL_FILES[mname].replace(".pkl", "")
            cm_file = os.path.join(OUTPUTS_DIR, f"cm_{safe}.png")
        if os.path.exists(cm_file):
            cm_cols[i % 3].image(cm_file, caption=mname, use_container_width=True)

# ──────────────────────────────────────────
# TAB 3 — ABOUT
# ──────────────────────────────────────────
with tab_about:
    st.header("About This Application")
    st.markdown("""
### ASD Screening in Children using Machine Learning & ANN

This Streamlit application predicts the likelihood of **Autism Spectrum Disorder (ASD)**
in children based on the AQ-10 screening questionnaire and basic demographic information.

#### Models Implemented
| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| Random Forest | Ensemble |
| K-Nearest Neighbors (KNN) | Instance-based |
| SVM — Linear kernel | Kernel-based |
| SVM — RBF kernel | Kernel-based |
| Naive Bayes | Probabilistic |
| Linear Discriminant Analysis (LDA) | Dimensionality reduction |
| Artificial Neural Network (ANN) | Deep Learning |

#### Training Pipeline (local — train.py)
1. Load and clean dataset  
2. Label encode categorical features  
3. 80/20 stratified train-test split  
4. SMOTE oversampling on training set  
5. StandardScaler feature normalisation  
6. Train all 9 models and evaluate  
7. Save models, scaler, encoders to `models/`  
8. Save all plots and metrics to `outputs/`

#### Input Features
| Feature | Description |
|---|---|
| A1 – A10 | AQ-10 behavioural screening questions (0 / 1) |
| Age | Child's age in years |
| Sex | Male / Female |
| Jaundice | Born with jaundice (Yes / No) |
| Family ASD | Family history of ASD (Yes / No) |

#### ⚠️ Disclaimer
This application is for **screening and educational purposes only**.  
It is **not** a substitute for professional medical diagnosis.  
Always consult a qualified healthcare professional.
    """)
