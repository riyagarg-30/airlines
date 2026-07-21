"""
Shared logic used by both train_model.py and app.py.
Kept in one place so the two never drift out of sync.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Keyword groups used to tag which service theme a review is about.
# This doubles as the "topic model" — transparent, fast, no heavy deps,
# and easy for a non-technical reviewer to audit (unlike a black-box LDA).
THEME_KEYWORDS = {
    "Flight Delays": ["delay", "late", "tarmac", "on time", "departure", "missed my connect"],
    "Baggage Handling": ["baggage", "luggage", "bag ", "suitcase"],
    "Cabin Crew Service": ["crew", "staff", "attendant", "flight attendants", "hostess"],
    "Seat Comfort": ["seat", "legroom", "recline", "cramped", "cushion"],
    "Food & Catering": ["food", "meal", "breakfast", "vegetarian", "catering"],
    "WiFi & Entertainment": ["wifi", "wi-fi", "entertainment", "ife", "screen"],
    "Price & Value": ["price", "value", "overpriced", "fee", "money"],
    "Check-in & Ground Staff": ["check-in", "check in", "kiosk", "counter", "ground staff", "queue"],
}

RECOMMENDATIONS = {
    "Flight Delays": "Improve delay communication and proactive rebooking — notify passengers the moment a delay is confirmed and auto-offer rebooking for missed connections.",
    "Baggage Handling": "Audit baggage handling at transfer hubs and add real-time bag tracking in the app.",
    "Cabin Crew Service": "Reinforce service-recovery training for cabin crew, especially on long-haul economy sectors.",
    "Seat Comfort": "Prioritize cabin refurbishment on older long-haul aircraft; flag seat-comfort complaints to fleet scheduling.",
    "Food & Catering": "Refresh catering partners on long-haul routes and expand special-meal stock to avoid run-outs.",
    "WiFi & Entertainment": "Fast-track IFE hardware refresh on aircraft with reported outages and audit wifi vendor uptime.",
    "Price & Value": "Review ancillary fee transparency and consider bundling seat selection for loyalty-tier members.",
    "Check-in & Ground Staff": "Increase check-in counter staffing at peak hours and maintain self-service kiosks proactively.",
    "General": "No single service theme stood out — keep monitoring for recurring language in future reviews.",
}


def get_sentiment(text: str) -> dict:
    """Returns VADER compound score (-1..1) plus a friendly label."""
    scores = _analyzer.polarity_scores(text or "")
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"compound": compound, "label": label}


def detect_themes(text: str, top_n: int = 2) -> list:
    """Returns the top matching service themes for a review, by keyword hit count."""
    text_lower = (text or "").lower()
    hits = {}
    for theme, keywords in THEME_KEYWORDS.items():
        count = sum(text_lower.count(k) for k in keywords)
        if count > 0:
            hits[theme] = count
    if not hits:
        return ["General"]
    ranked = sorted(hits, key=hits.get, reverse=True)
    return ranked[:top_n]


def recommend_for_themes(themes: list) -> list:
    return [RECOMMENDATIONS.get(t, RECOMMENDATIONS["General"]) for t in themes]


def satisfaction_bucket(score_0_100: float) -> str:
    if score_0_100 >= 70:
        return "High Satisfaction"
    elif score_0_100 >= 45:
        return "Moderate Satisfaction"
    else:
        return "Low Satisfaction"
