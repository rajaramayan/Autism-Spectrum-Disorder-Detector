# ==========================================
# TRAIN.PY — LOCAL TRAINING SCRIPT
# ==========================================
# Run this script LOCALLY in VS Code to:
#   - Train all ML models + ANN
#   - Save trained models to models/
#   - Save scaler and encoders to models/
#   - Generate and save all output plots to outputs/
#
# DO NOT run this script on Render / online.
# ==========================================

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves to file, no pop-up
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, log_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================
# PATHS
# ==========================================
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"
DATA_PATH   = "Autism_Screening_Data_Combined.csv"

# Consistent filename mapping for all classical models
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

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("=" * 50)
print("STEP 1 — Loading dataset")
print("=" * 50)
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.dtypes)

# ==========================================
# 2. PREPROCESSING
# ==========================================
print("\nSTEP 2 — Preprocessing")
df.drop_duplicates(inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Identify categorical columns (includes target 'Class')
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

# Encode each categorical column individually and save encoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Save ONLY input-feature encoders (exclude target 'Class')
input_encoders = {col: enc for col, enc in label_encoders.items() if col != "Class"}
joblib.dump(input_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
print("Label encoders saved.")

# Features (X) and target (y)
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column = Class

feature_names = X.columns.tolist()
joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
print(f"Features ({len(feature_names)}): {feature_names}")

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================
print("\nSTEP 3 — Train-test split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================================
# 4. HANDLE IMBALANCE (SMOTE)
# ==========================================
print("\nSTEP 4 — SMOTE oversampling")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Train: {X_train.shape}")

# ==========================================
# 5. FEATURE SCALING
# ==========================================
print("\nSTEP 5 — Feature scaling")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("Scaler saved.")

# ==========================================
# 6. TRAIN + EVALUATE CLASSICAL MODELS
# ==========================================
print("\nSTEP 6 — Training classical ML models")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(),
    "Random Forest":       RandomForestClassifier(),
    "KNN":                 KNeighborsClassifier(),
    "SVM (Linear)":        SVC(kernel='linear', probability=True),
    "SVM (RBF)":           SVC(kernel='rbf',    probability=True),
    "Naive Bayes":         GaussianNB(),
    "LDA":                 LinearDiscriminantAnalysis(),
}

results  = []
roc_data = []

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc      = accuracy_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred, zero_division=0)
    rec      = recall_score(y_test, y_pred, zero_division=0)
    f1       = f1_score(y_test, y_pred, zero_division=0)
    roc_auc  = roc_auc_score(y_test, y_prob)
    logloss  = log_loss(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    results.append([name, acc, prec, rec, specificity, f1, roc_auc, logloss])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data.append((name, fpr, tpr, roc_auc))

    # Save model
    filename = MODEL_FILES[name]
    joblib.dump(model, os.path.join(MODELS_DIR, filename))
    print(f"    Saved  → models/{filename}")

# ==========================================
# 7. ANN MODEL
# ==========================================
print("\nSTEP 7 — Training ANN")
n_features = X_train_scaled.shape[1]

ann = Sequential([
    Dense(32, activation='relu', input_shape=(n_features,)),
    Dense(16, activation='relu'),
    Dense(1,  activation='sigmoid'),
])
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = ann.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
)

ann.save(os.path.join(MODELS_DIR, "ann_model.h5"))
print("ANN model saved → models/ann_model.h5")

# ANN Evaluation
y_prob_ann = ann.predict(X_test_scaled).flatten()
y_pred_ann = (y_prob_ann > 0.5).astype(int)

acc_ann      = accuracy_score(y_test, y_pred_ann)
prec_ann     = precision_score(y_test, y_pred_ann, zero_division=0)
rec_ann      = recall_score(y_test, y_pred_ann, zero_division=0)
f1_ann       = f1_score(y_test, y_pred_ann, zero_division=0)
roc_auc_ann  = roc_auc_score(y_test, y_prob_ann)
logloss_ann  = log_loss(y_test, y_prob_ann)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ann).ravel()
specificity_ann = tn / (tn + fp) if (tn + fp) > 0 else 0.0

results.append(["ANN", acc_ann, prec_ann, rec_ann, specificity_ann,
                f1_ann, roc_auc_ann, logloss_ann])

fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)
roc_data.append(("ANN", fpr_ann, tpr_ann, roc_auc_ann))

# ==========================================
# 8. RESULTS DATAFRAME
# ==========================================
columns = ["Model", "Accuracy", "Precision", "Recall",
           "Specificity", "F1 Score", "ROC-AUC", "Log Loss"]
results_df = pd.DataFrame(results, columns=columns)
results_df_sorted = results_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

results_df_sorted.to_csv(os.path.join(OUTPUTS_DIR, "model_results.csv"), index=False)
print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)
print(results_df_sorted.to_string(index=False))

# Save best model name
best_model_name = results_df_sorted.iloc[0]["Model"]
joblib.dump(best_model_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))
print(f"\nBest model: {best_model_name}")

# ==========================================
# 9. GENERATE & SAVE OUTPUT PLOTS
# ==========================================
print("\nSTEP 9 — Saving output plots")

# --- ROC Curves (all models) ---
plt.figure(figsize=(10, 8))
for name, fpr, tpr, auc in roc_data:
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison — All Models Including ANN")
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "roc_curves.png"), dpi=150)
plt.close()
print("  Saved → outputs/roc_curves.png")

# --- Bar Chart (Accuracy / F1 / ROC-AUC) ---
fig, ax = plt.subplots(figsize=(13, 6))
results_df_sorted.set_index("Model")[["Accuracy", "F1 Score", "ROC-AUC"]].plot(kind='bar', ax=ax)
ax.set_title("Model Performance Comparison")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "model_comparison.png"), dpi=150)
plt.close()
print("  Saved → outputs/model_comparison.png")

# --- Confusion Matrix — Best Model ---
if best_model_name == "ANN":
    y_pred_best = y_pred_ann
else:
    best_model_obj = joblib.load(os.path.join(MODELS_DIR, MODEL_FILES[best_model_name]))
    y_pred_best = best_model_obj.predict(X_test_scaled)

cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No ASD', 'ASD'],
            yticklabels=['No ASD', 'ASD'])
plt.title(f"Confusion Matrix — {best_model_name} (Best)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix_best.png"), dpi=150)
plt.close()
print("  Saved → outputs/confusion_matrix_best.png")

# --- ANN Training History ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title("ANN Accuracy per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title("ANN Loss per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "ann_training_history.png"), dpi=150)
plt.close()
print("  Saved → outputs/ann_training_history.png")

# --- Individual Confusion Matrices (all classical models) ---
for name, model_obj in models.items():
    y_pred_m = model_obj.predict(X_test_scaled)
    cm_m = confusion_matrix(y_test, y_pred_m)
    safe  = MODEL_FILES[name].replace(".pkl", "")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_m, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No ASD', 'ASD'],
                yticklabels=['No ASD', 'ASD'])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"cm_{safe}.png"), dpi=150)
    plt.close()

# ANN confusion matrix
cm_ann = confusion_matrix(y_test, y_pred_ann)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No ASD', 'ASD'],
            yticklabels=['No ASD', 'ASD'])
plt.title("Confusion Matrix — ANN")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "cm_ANN.png"), dpi=150)
plt.close()
print("  Saved individual confusion matrices for all models.")

# ==========================================
# DONE
# ==========================================
print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)
print(f"  Models saved to : {MODELS_DIR}/")
print(f"  Outputs saved to: {OUTPUTS_DIR}/")
print("\nNext steps:")
print("  1. Commit models/ and outputs/ to your Git repository.")
print("  2. Push to GitHub.")
print("  3. Deploy app.py on Render.")
