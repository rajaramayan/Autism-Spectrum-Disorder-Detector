# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("Autism_Screening_Data_Combined.csv")

print("Shape:", df.shape)
print(df.info())
print(df.describe())

# ==========================================
# 3. PREPROCESSING
# ==========================================
df.drop_duplicates(inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 5. HANDLE IMBALANCE (SMOTE)
# ==========================================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ==========================================
# 6. FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 7. DEFINE MODELS
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM (Linear)": SVC(kernel='linear', probability=True),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis()
}

# ==========================================
# 8. TRAIN + EVALUATE CLASSICAL MODELS
# ==========================================
results = []
roc_data = []   # store ROC info

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    results.append([name, acc, prec, rec, specificity, f1, roc_auc, logloss])

    # Store ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data.append((name, fpr, tpr, roc_auc))

# ==========================================
# 9. CREATE RESULTS DATAFRAME
# ==========================================
columns = ["Model", "Accuracy", "Precision", "Recall",
           "Specificity", "F1 Score", "ROC-AUC", "Log Loss"]

results_df = pd.DataFrame(results, columns=columns)

# ==========================================
# 10. ANN MODEL
# ==========================================
ann = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = ann.fit(X_train_scaled, y_train,
                  epochs=50,
                  batch_size=16,
                  validation_split=0.2,
                  verbose=0)

# ANN Evaluation
y_prob_ann = ann.predict(X_test_scaled).flatten()
y_pred_ann = (y_prob_ann > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred_ann)
prec = precision_score(y_test, y_pred_ann)
rec = recall_score(y_test, y_pred_ann)
f1 = f1_score(y_test, y_pred_ann)
roc_auc = roc_auc_score(y_test, y_prob_ann)
logloss = log_loss(y_test, y_prob_ann)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ann).ravel()
specificity = tn / (tn + fp)

# Add ANN results
results_df.loc[len(results_df)] = [
    "ANN", acc, prec, rec, specificity, f1, roc_auc, logloss
]

# Store ANN ROC
fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)
roc_data.append(("ANN", fpr_ann, tpr_ann, roc_auc))

print("\nFinal Model Comparison:\n", results_df)

# ==========================================
# 11. ROC CURVE (ALL MODELS INCLUDING ANN)
# ==========================================
plt.figure(figsize=(10, 8))

for name, fpr, tpr, auc in roc_data:
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (All Models Including ANN)")
plt.legend()
plt.show()

# ==========================================
# 12. BAR PLOT (ALL MODELS INCLUDING ANN)
# ==========================================
results_df_sorted = results_df.sort_values(by="ROC-AUC", ascending=False)

results_df_sorted.set_index("Model")[["Accuracy", "F1 Score", "ROC-AUC"]].plot(kind='bar')

plt.title("Model Performance Comparison (All Models)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# 13. CONFUSION MATRIX (BEST MODEL)
# ==========================================
best_model_name = results_df_sorted.iloc[0]["Model"]

if best_model_name == "ANN":
    y_pred_best = y_pred_ann
else:
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best)

sns.heatmap(cm, annot=True, fmt='d')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

# ==========================================
# 14. FINAL BEST MODEL
# ==========================================
best_model_final = results_df_sorted.iloc[0]

print("\nBEST MODEL:")
print(best_model_final)