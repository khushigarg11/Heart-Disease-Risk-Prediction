## Heart-Disease-Risk-Prediction
An end-to-end machine learning project that analyzes clinical patient data, 
benchmarks five classification algorithms to predict heart disease, 
and deploys the best-performing model as an interactive Streamlit web app for real-time risk prediction.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [The Web App](#the-web-app)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)

---

## Overview

This project takes a raw clinical dataset through EDA, data-quality
correction, encoding, and a five-model comparison to find the best
classifier for predicting heart disease — then ships that model in a
Streamlit app where a user can enter their own health details and get an
instant risk prediction.

> **Headline result:** KNN was the top performer at **88.6% accuracy**
> (F1 = 0.899), narrowly ahead of Logistic Regression (87.5%), and is the
> model deployed in the live app.

**Live demo:** https://heart-disease-prediction-app-45xjpwztphqm2d82ogs6cz.streamlit.app/

## Dataset

| Property | Value |
|---|---|
| File | `heart.csv` |
| Rows | 918 |
| Features | 11 predictors + 1 target |
| Target | `HeartDisease` (binary) |
| Duplicate rows | 0 |
| Missing values | 0 (but see data-quality note below) |

| Column | Type | Description |
|---|---|---|
| `Age` | numeric | Patient age |
| `Sex` | categorical | `M` / `F` |
| `ChestPainType` | categorical | `TA`, `ATA`, `NAP`, `ASY` |
| `RestingBP` | numeric | Resting blood pressure (mm Hg) |
| `Cholesterol` | numeric | Serum cholesterol (mg/dL) |
| `FastingBS` | binary | Fasting blood sugar > 120 mg/dL |
| `RestingECG` | categorical | `Normal`, `ST`, `LVH` |
| `MaxHR` | numeric | Maximum heart rate achieved |
| `ExerciseAngina` | categorical | `Y` / `N` |
| `Oldpeak` | numeric | ST depression from exercise |
| `ST_Slope` | categorical | `Up`, `Flat`, `Down` |
| `HeartDisease` | binary | **Target** |

**Data-quality correction:** `Cholesterol` and `RestingBP` both contained
physiologically invalid `0` entries. Each was replaced with the mean of its
own non-zero values before modeling — a necessary fix, since a cholesterol
or blood pressure reading of zero is not clinically possible.

## Methodology

**1. Exploratory Data Analysis**
Reviewed shape, dtypes, and summary statistics; confirmed zero duplicates
and zero nulls. Plotted target class balance, histograms of numeric
features, and count plots of `ChestPainType` and `FastingBS` segmented by
`HeartDisease`. Compared `Cholesterol` and `Age` across disease/no-disease
groups via box and violin plots, and reviewed a correlation heatmap.

**2. Data-Quality Correction**
Replaced invalid `Cholesterol == 0` and `RestingBP == 0` entries with the
mean of their respective non-zero values.

**3. Encoding**
One-hot encoded all categorical fields with `pd.get_dummies(drop_first=True)`
and cast the result to integer type.

**4. Model Comparison**
Split data 80/20 with stratification on the target (`random_state=42`),
scaled features with `StandardScaler`, then trained and evaluated five
classifiers under identical conditions:

- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Decision Tree
- SVM (RBF kernel)

**5. Model Export**
Serialized the winning model (KNN), the fitted scaler, and the expected
column order with `joblib` for use in the deployed app.

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| **KNN** | **0.886** | **0.899** |
| Logistic Regression | 0.875 | 0.888 |
| SVM (RBF Kernel) | 0.864 | 0.880 |
| Naive Bayes | 0.870 | 0.879 |
| Decision Tree | 0.761 | 0.778 |

## The Web App

`app.py` is a **Streamlit** application that puts the trained KNN model in
front of an end user, and is live at:

**https://heart-disease-prediction-app-45xjpwztphqm2d82ogs6cz.streamlit.app/**

It works as follows:

- Collects 11 patient inputs through sliders, number inputs, and dropdowns
  (age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar,
  resting ECG, max heart rate, exercise-induced angina, oldpeak, ST slope).
- Reconstructs the same one-hot encoded feature layout used in training by
  building a single-row DataFrame, filling any missing dummy columns with
  0, and reordering columns to match `columns.pkl` exactly.
- Scales the input with the saved `scaler.pkl` and predicts with the saved
  `KNN_heart.pkl`.
- Displays a clear **High Risk** or **Low Risk** result with a plain-language
  message, explicitly recommending professional medical consultation rather
  than treating the prediction as a diagnosis.

**Files the app depends on (produced by the notebook):**

| File | Produced by | Purpose |
|---|---|---|
| `KNN_heart.pkl` | `joblib.dump(models["KNN"], ...)` | Trained KNN classifier |
| `scaler.pkl` | `joblib.dump(scaler, ...)` | Fitted `StandardScaler` |
| `columns.pkl` | `joblib.dump(X.columns.tolist(), ...)` | Expected feature column order |

## Key Insights

- **KNN outperforming Logistic Regression and SVM** suggests the decision
  boundary between disease/no-disease has some local, non-linear structure
  that instance-based methods capture slightly better here.
- **Decision Tree trails noticeably behind every other model** (76.1% vs.
  86–89% for the top three), indicating it likely overfit the training
  split without pruning or depth constraints.
- The **Cholesterol/RestingBP zero-value fix** was essential — without it,
  a meaningful chunk of records would have carried a false signal that
  could mislead any of the five models.
- The deployed app's **input reconstruction logic must stay in sync with
  the training encoding**: any change to categorical value options during
  retraining requires regenerating `columns.pkl`, or the app's column
  alignment will silently break.

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · joblib · Streamlit

## Project Structure
```
.
├── HeartDiseasePrediction.ipynb   # EDA, cleaning, model comparison, model export
├── heart.csv                       # Raw dataset
├── app.py                          # Streamlit web app
├── KNN_heart.pkl                   # Trained model (generated by the notebook)
├── scaler.pkl                      # Fitted StandardScaler (generated by the notebook)
├── columns.pkl                     # Expected feature columns (generated by the notebook)
└── README.md                       # This file
```

## Setup & Usage

The app is already deployed and usable directly at
**https://heart-disease-prediction-app-45xjpwztphqm2d82ogs6cz.streamlit.app/**
— no setup needed. To run it locally instead:

**1. Run the notebook first** to generate the model artifacts:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook HeartDiseasePrediction.ipynb
```
Ensure `heart.csv` is in the same directory. Running all cells produces
`KNN_heart.pkl`, `scaler.pkl`, and `columns.pkl`.

**2. Launch the web app:**
```bash
pip install streamlit
streamlit run app.py
```
The app requires `KNN_heart.pkl`, `scaler.pkl`, and `columns.pkl` to be
present in the same directory as `app.py`.
