# Flight Feedback Intelligence

A deployable app that predicts customer satisfaction from airline review text,
detects the service theme driving the score (delays, baggage, crew, seat comfort,
food, wifi/entertainment, price, check-in), and generates a targeted recommendation.

## What's inside (kept intentionally minimal)

```
app.py                    Streamlit UI (single file — review analyzer + dashboard tab)
train_model.py             Trains the satisfaction model, run this to (re)build models/
utils.py                    Shared sentiment + theme-detection logic used by both
requirements.txt            Pinned dependencies
data/reviews.csv             Sample review dataset (see note below)
models/
  tfidf_vectorizer.pkl        Fitted TF-IDF vectorizer
  csat_model.pkl               Trained RandomForestRegressor (predicts 1–10 rating)
  meta.pkl                      Test-set MAE/R² for the dashboard's "model performance" panel
```

## About the dataset

The original British Airways review dataset was a Google Drive file that
required sign-in to fetch, which wasn't reachable from where this was built.
`data/reviews.csv` is a **realistic sample dataset** (260 reviews) covering the
same themes real BA reviews cover — delays, baggage, crew, seat comfort, food,
wifi, price, check-in — so the whole pipeline is trained, tested, and working
out of the box.

**To use your real dataset:** replace `data/reviews.csv` with your export,
keeping (or remapping in `train_model.py` → `COLUMN_MAP`) two columns:
- `review_text` — the free-text review
- `rating` — a numeric score (any scale works, just update the 1–10 references)

Then re-run:
```
python train_model.py
```
This overwrites the files in `models/` with a model trained on your real data.
No other file needs to change.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (all files, including `models/*.pkl` and `data/reviews.csv`)
2. Go to https://share.streamlit.io → "New app"
3. Point it at the repo, branch, and `app.py` as the entry file
4. Deploy — no extra config needed, `requirements.txt` is picked up automatically

## How it works

- **Sentiment**: VADER (`vaderSentiment`) — rule-based, no model download needed,
  fast enough for real-time use.
- **Satisfaction prediction**: TF-IDF (300 features, unigrams+bigrams) + sentiment
  score → `RandomForestRegressor` predicting a 1–10 rating, shown in-app as a 0–100 score.
- **Theme detection**: transparent keyword-group matching (in `utils.py`) instead
  of a black-box topic model — easy to audit and to extend with new keywords,
  and doesn't need extra heavy dependencies (gensim/BERTopic) for deployment.
- **Recommendations**: each theme maps to one specific, actionable recommendation,
  also editable in `utils.py` → `RECOMMENDATIONS`.

## Extending this later

- Swap `RandomForestRegressor` for a cloud AutoML endpoint (SageMaker/Vertex/Azure ML)
  by replacing `predict_rating()` in `app.py` with an API call.
- Swap keyword-based theming for BERTopic/LDA once you have enough real reviews
  for a topic model to be stable (roughly 1,000+ documents).
- Wrap `predict_rating()` behind a FastAPI/Lambda endpoint if you need it
  outside of Streamlit too.
