"""
Trains the customer satisfaction (CSAT) prediction model.

Run it with:  python train_model.py
Reads:        data/reviews.csv   (columns: review_text, route, cabin_class, rating, recommended)
Writes:       models/tfidf_vectorizer.pkl, models/csat_model.pkl, models/meta.pkl

To use your REAL British Airways dataset instead of the bundled sample:
just replace data/reviews.csv with your file (keep the same column names,
or edit COLUMN_MAP below to match your file) and re-run this script.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from utils import get_sentiment

# If your real dataset uses different column names, map them here:
COLUMN_MAP = {
    "review_text": "review_text",
    "rating": "rating",
}

DATA_PATH = "data/reviews.csv"


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items() if v in df.columns})
    df = df.dropna(subset=["review_text", "rating"])
    print(f"  {len(df)} reviews loaded")

    print("Scoring sentiment for every review (VADER) ...")
    sentiment_scores = df["review_text"].apply(lambda t: get_sentiment(t)["compound"])
    df["sentiment_compound"] = sentiment_scores

    print("Vectorizing review text (TF-IDF) ...")
    vectorizer = TfidfVectorizer(max_features=300, stop_words="english", ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df["review_text"]).toarray()

    X = np.hstack([X_text, df[["sentiment_compound"]].values])
    y = df["rating"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForestRegressor ...")
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"  Test MAE: {mae:.2f} (rating scale 1-10)")
    print(f"  Test R^2: {r2:.2f}")

    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(model, "models/csat_model.pkl")
    joblib.dump({"mae": mae, "r2": r2, "n_train": len(X_train)}, "models/meta.pkl")
    print("Saved models/tfidf_vectorizer.pkl, models/csat_model.pkl, models/meta.pkl")


if __name__ == "__main__":
    main()
