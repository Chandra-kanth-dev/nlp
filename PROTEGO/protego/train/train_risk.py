
"""
train_risk.py
=============
Training script for risk classification model in PROTEGO.

Predicts:
- low
- medium
- high
- emergency

This model is SAFETY-CRITICAL and is later fused with:
- emotion analysis
- sentiment analysis
- linguistic features
- rule-based overrides

Design goals:
- Conservative risk detection
- Reproducible training
- Runtime compatibility
- Audit-ready outputs
"""

import pandas as pd
import joblib
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from protego.nlp.preprocess import clean_text


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "risk_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# Load & validate dataset
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)

required_cols = {"text", "risk"}
if not required_cols.issubset(df.columns):
    raise ValueError("Dataset must contain 'text' and 'risk' columns")

# Drop invalid rows
df = df.dropna(subset=["text", "risk"])

# Normalize labels
df["risk"] = df["risk"].str.strip().str.lower()

ALLOWED_RISKS = {"low", "medium", "high", "emergency"}
invalid_labels = set(df["risk"]) - ALLOWED_RISKS
if invalid_labels:
    raise ValueError(f"Invalid risk labels found: {invalid_labels}")

print("📊 Risk label distribution:")
print(Counter(df["risk"]))
print("-" * 40)


# -------------------------------------------------
# Preprocess text
# -------------------------------------------------
df["clean_text"] = df["text"].astype(str).apply(clean_text)

X = df["clean_text"]
y = df["risk"]


# -------------------------------------------------
# Train-test split (stratified)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------------------------
# TF-IDF Vectorization
# -------------------------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=1
)


X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -------------------------------------------------
# Train model (safety-biased)
# -------------------------------------------------
model = MultinomialNB(
    alpha=0.7  # stronger smoothing to avoid under-predicting emergencies
)

model.fit(X_train_vec, y_train)


# -------------------------------------------------
# Evaluate model
# -------------------------------------------------
y_pred = model.predict(X_test_vec)

print("\n📊 Risk Model Evaluation")
print("-" * 40)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# Save artifacts (RUNTIME COMPATIBLE)
# -------------------------------------------------
joblib.dump(model, MODEL_DIR / "risk_model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "risk_vectorizer.pkl")

print("\n✅ Risk model training complete")
print(f"📁 Model saved to: {MODEL_DIR / 'risk_model.pkl'}")
print(f"📁 Vectorizer saved to: {MODEL_DIR / 'vectorizer.pkl'}")
