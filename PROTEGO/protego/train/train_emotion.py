"""
train_emotion.py
================
Training script for emotion classification in PROTEGO.

Pipeline:
- Load & validate dataset
- Apply deterministic preprocessing
- TF-IDF vectorization
- Logistic Regression (balanced)
- Evaluation & reporting
- Save model + vectorizer (runtime-compatible)
"""

import pandas as pd
import joblib
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from protego.nlp.preprocess import clean_text


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "emotion_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# Load & validate dataset
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)

required_cols = {"text", "emotion"}
if not required_cols.issubset(df.columns):
    raise ValueError("Dataset must contain 'text' and 'emotion' columns")

# Drop empty rows safely
df = df.dropna(subset=["text", "emotion"])

# Normalize labels
df["emotion"] = df["emotion"].str.strip().str.lower()

ALLOWED_EMOTIONS = {"sadness", "fear", "anger", "shame", "neutral"}
invalid_labels = set(df["emotion"]) - ALLOWED_EMOTIONS
if invalid_labels:
    raise ValueError(f"Invalid emotion labels found: {invalid_labels}")

print("📊 Emotion label distribution:")
print(Counter(df["emotion"]))
print("-" * 40)


# -------------------------------------------------
# Preprocess text
# -------------------------------------------------
df["clean_text"] = df["text"].astype(str).apply(clean_text)

X = df["clean_text"]
y = df["emotion"]


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
# Train model
# -------------------------------------------------
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_vec, y_train)


# -------------------------------------------------
# Evaluate model
# -------------------------------------------------
y_pred = model.predict(X_test_vec)

print("\n📊 Emotion Model Evaluation")
print("-" * 40)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# Save artifacts (RUNTIME COMPATIBLE)
# -------------------------------------------------
joblib.dump(model, MODEL_DIR / "emotion_model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "emotion_vectorizer.pkl")


print("\n✅ Emotion model training complete")
print(f"📁 Model saved to: {MODEL_DIR / 'emotion_model.pkl'}")
print(f"📁 Vectorizer saved to: {MODEL_DIR / 'vectorizer.pkl'}")
