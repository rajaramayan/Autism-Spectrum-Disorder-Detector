# ASD Screening in Children — Machine Learning & ANN

A Streamlit web application that predicts the likelihood of **Autism Spectrum Disorder (ASD)** in children using 9 pre-trained Machine Learning models and an Artificial Neural Network.

---

## Project Structure

```
project/
├── train.py                          ← Run LOCALLY to train & save all models
├── app.py                            ← Streamlit app (deployment / prediction only)
├── requirements.txt                  ← Render deployment dependencies
├── render.yaml                       ← Render service configuration
├── .gitignore
├── Autism_Screening_Data_Combined.csv
├── models/                           ← Auto-created by train.py
│   ├── Logistic_Regression.pkl
│   ├── Decision_Tree.pkl
│   ├── Random_Forest.pkl
│   ├── KNN.pkl
│   ├── SVM_Linear.pkl
│   ├── SVM_RBF.pkl
│   ├── Naive_Bayes.pkl
│   ├── LDA.pkl
│   ├── ann_model.keras
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   └── best_model_name.pkl
└── outputs/                          ← Auto-created by train.py
    ├── model_results.csv
    ├── roc_curves.png
    ├── model_comparison.png
    ├── confusion_matrix_best.png
    ├── ann_training_history.png
    └── cm_*.png  (one per model)
```

---

## Running Locally (VS Code)

### Step 1 — Open terminal in project folder

In VS Code press **Ctrl + `** to open the terminal, then run:

```powershell
cd "e:\Downloads\Chhaya Code deployment new"
```

---

### Step 2 — Create a virtual environment (recommended)

```powershell
python -m venv venv
venv\Scripts\activate
```

---

### Step 3 — Install all dependencies

```powershell
pip install scikit-learn imbalanced-learn tensorflow matplotlib seaborn joblib pandas numpy streamlit
```

---

### Step 4 — Train all models locally

python train.py```powershell
python train.py
```

This will:
- Train 8 classical ML models + ANN
- Save all trained models to `models/`
- Save all output plots and metrics to `outputs/`

> This step only needs to be run **once** (or whenever you retrain on new data).

---

### Step 5 — Run the Streamlit app locally

```powershell
streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**

---

### What to verify locally

| Tab | What to check |
|---|---|
| **Predict** | Fill the form and click "Run Prediction" — all 9 models return results |
| **Model Performance** | Metrics table, ROC curves, and confusion matrices display correctly |
| **About** | Project description renders properly |

---

## Deploying to Render

Once you are satisfied with local testing, follow these steps.

### Step 1 — Push your project to GitHub

```powershell
git init
git add .
git commit -m "Initial deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

> **Important:** Make sure the `models/` and `outputs/` folders are committed.  
> Render reads these files at runtime — without them the app will not work.

---

### Step 2 — Create a Web Service on Render

1. Go to [https://render.com](https://render.com) and sign in.
2. Click **New → Web Service**.
3. Connect your GitHub account and select your repository.
4. Render will auto-detect `render.yaml` and fill in settings automatically.

If you prefer to set them manually:

| Setting | Value |
|---|---|
| **Environment** | Python |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| **Python Version** | `3.10.0` |

5. Click **Create Web Service**.
6. Render will build and deploy your app. Once complete, a public URL is provided.

---

## Models Used

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

---

## Input Features

| Feature | Description |
|---|---|
| A1 – A10 | AQ-10 behavioural screening questions (0 = No, 1 = Yes) |
| Age | Child's age in years (1–18) |
| Sex | Male / Female |
| Jaundice | Born with jaundice (Yes / No) |
| Family ASD | Family history of ASD (Yes / No) |

---

## Disclaimer

> This application is for **screening and educational purposes only**.  
> It is **not** a substitute for professional medical diagnosis.  
> Always consult a qualified healthcare professional.
