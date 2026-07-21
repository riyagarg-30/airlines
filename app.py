

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils import (
    get_sentiment,
    detect_themes,
    recommend_for_themes,
    satisfaction_bucket,
)

st.set_page_config(
    page_title="Flight Feedback Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html,body,[class*="css"]{
font-family:Poppins,sans-serif;
}

.stApp{
background:
radial-gradient(circle at top left,#2449ff33,transparent 35%),
radial-gradient(circle at bottom right,#8b5cf633,transparent 35%),
linear-gradient(135deg,#081120,#0B1730,#081120);
background-size:300% 300%;
animation:bgmove 18s ease infinite;
color:white;
}

@keyframes bgmove{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

header,footer,#MainMenu{visibility:hidden;}

.block-container{
padding-top:2rem;
padding-left:3rem;
padding-right:3rem;
}

.hero{
padding:38px;
border-radius:28px;
background:rgba(255,255,255,.07);
backdrop-filter:blur(20px);
border:1px solid rgba(255,255,255,.12);
box-shadow:0 10px 40px rgba(0,0,0,.35);
}

.hero h1{
font-family:'Space Grotesk',sans-serif;
font-size:48px;
margin-bottom:8px;
}

.hero p{
font-size:18px;
color:#d6e3ff;
}

.metric-card{
background:rgba(255,255,255,.06);
backdrop-filter:blur(18px);
padding:22px;
border-radius:22px;
border:1px solid rgba(255,255,255,.10);
transition:.35s;
}

.metric-card:hover{
transform:translateY(-6px);
box-shadow:0 15px 35px rgba(0,0,0,.4);
}

.stButton>button{
height:52px;
border-radius:14px;
border:none;
font-weight:600;
background:linear-gradient(90deg,#3b82f6,#7c3aed);
color:white;
}

.stButton>button:hover{
transform:scale(1.02);
}

textarea{
border-radius:18px!important;
background:rgba(255,255,255,.05)!important;
color:white!important;
}

.badge{
display:inline-block;
padding:8px 15px;
margin:4px;
border-radius:999px;
background:#2563eb;
color:white;
font-size:14px;
}

.rec{
background:rgba(255,255,255,.05);
border-left:4px solid #38bdf8;
padding:16px;
margin-bottom:10px;
border-radius:10px;
}

hr{
border-color:rgba(255,255,255,.1);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/csat_model.pkl")
    meta = joblib.load("models/meta.pkl")
    return vectorizer, model, meta

@st.cache_data
def load_reference_data():
    return pd.read_csv("data/reviews.csv")

vectorizer, model, meta = load_artifacts()

def predict_rating(text):
    sentiment = get_sentiment(text)["compound"]
    X_text = vectorizer.transform([text]).toarray()
    X = np.hstack([X_text, [[sentiment]]])
    pred = model.predict(X)[0]
    return float(np.clip(pred,1,10))

# ---------- HERO ----------
left,right = st.columns([5,1])

with left:
    st.markdown("""
    <div class='hero'>
    <h1>✈ Flight Feedback Intelligence</h1>
    <p>
    AI powered airline review analytics.
    Predict customer satisfaction, discover themes,
    and generate business recommendations.
    </p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("<h1 style='font-size:72px;text-align:center;'>✈️</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    "🔍 Analyze Review",
    "📊 Dashboard"
])



with tab1:

    st.markdown("## ✨ AI Review Analyzer")
    st.caption("Paste a customer review and let AI analyze the experience.")

    example = (
        "The flight was delayed by four hours. "
        "Cabin crew were polite but communication "
        "was poor and baggage arrived late."
    )

    left,right = st.columns([5,1])

    with left:
        review_text = st.text_area(
            "Customer Review",
            value="",
            placeholder=example,
            height=220
        )

    with right:
        st.write("")
        st.write("")
        use_example = st.button("📄 Example", use_container_width=True)
        analyze = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if use_example:
        review_text = example
        analyze = True

    if analyze:

        if not review_text.strip():
            st.warning("Please enter a review.")
        else:

            with st.spinner("🤖 AI is analyzing the review..."):

                sentiment = get_sentiment(review_text)

                predicted = predict_rating(review_text)

                score = round((predicted-1)/9*100)

                bucket = satisfaction_bucket(score)

                themes = detect_themes(review_text,3)

                recommendations = recommend_for_themes(themes)

            st.markdown("---")

            c1,c2,c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                <h4>😊 Sentiment</h4>
                <h2>{sentiment["label"]}</h2>
                <p>Compound Score : {sentiment["compound"]:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                <h4>⭐ Satisfaction</h4>
                <h2>{score}/100</h2>
                <p>{bucket}</p>
                </div>
                """, unsafe_allow_html=True)

            with c3:

                chips = ""

                icon_map = {
                    "delay":"🛫",
                    "staff":"👨‍✈️",
                    "crew":"👨‍✈️",
                    "food":"🍽️",
                    "seat":"💺",
                    "comfort":"💺",
                    "service":"🤝",
                    "baggage":"🧳"
                }

                for t in themes:
                    icon="🏷️"
                    low=t.lower()
                    for k,v in icon_map.items():
                        if k in low:
                            icon=v
                            break
                    chips += f"<span class='badge'>{icon} {t}</span>"

                st.markdown(f"""
                <div class="metric-card">
                <h4>🏷️ Themes</h4>
                <br>
                {chips}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 💡 Recommended Actions")

            for rec in recommendations:
                st.markdown(
                    f"<div class='rec'>✅ {rec}</div>",
                    unsafe_allow_html=True
                )

            st.success("Analysis Complete ✔")


with tab2:

    df = load_reference_data()

    with st.spinner("Preparing analytics dashboard..."):

        df["sentiment"] = df["review_text"].apply(
            lambda x: get_sentiment(x)["label"]
        )

        df["theme"] = df["review_text"].apply(
            lambda x: detect_themes(x,1)[0]
        )

    st.markdown("## 📊 Portfolio Analytics Dashboard")

    k1,k2,k3,k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class='metric-card'>
        <h4>📝 Reviews</h4>
        <h2>{len(df)}</h2>
        </div>
        """,unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class='metric-card'>
        <h4>⭐ Avg Rating</h4>
        <h2>{df['rating'].mean():.1f}/10</h2>
        </div>
        """,unsafe_allow_html=True)

    with k3:
        positive=(df["sentiment"]=="Positive").mean()*100
        st.markdown(f"""
        <div class='metric-card'>
        <h4>😊 Positive</h4>
        <h2>{positive:.0f}%</h2>
        </div>
        """,unsafe_allow_html=True)

    with k4:
        recommend=(df["recommended"]=="yes").mean()*100
        st.markdown(f"""
        <div class='metric-card'>
        <h4>👍 Recommend</h4>
        <h2>{recommend:.0f}%</h2>
        </div>
        """,unsafe_allow_html=True)

    st.markdown("---")

    left,right = st.columns(2)

    with left:
        st.subheader("📈 Sentiment Distribution")
        st.bar_chart(df["sentiment"].value_counts())

    with right:
        st.subheader("🏷️ Service Themes")
        st.bar_chart(df["theme"].value_counts())

    st.markdown("---")

    st.subheader("📋 Sample Reviews")

    st.dataframe(
        df[
            [
                "review_text",
                "rating",
                "sentiment",
                "theme"
            ]
        ].head(15),
        use_container_width=True,
        height=360
    )

    with st.expander("🤖 Model Performance"):

        st.metric(
            "Mean Absolute Error",
            f"{meta['mae']:.2f}"
        )

        st.metric(
            "R² Score",
            f"{meta['r2']:.2f}"
        )

        st.info(
            "Replace data/reviews.csv with a larger dataset "
            "and retrain using train_model.py for production results."
        )

st.markdown("---")
st.caption("Premium Flight Feedback Intelligence Dashboard • Part 3")


import plotly.express as px
import plotly.graph_objects as go

st.markdown("""
<style>

.plot-card{
background:rgba(255,255,255,.06);
backdrop-filter:blur(18px);
padding:20px;
border-radius:22px;
border:1px solid rgba(255,255,255,.10);
margin-bottom:18px;
}

.footer-card{
margin-top:40px;
padding:25px;
text-align:center;
border-radius:22px;
background:linear-gradient(90deg,#2563eb33,#7c3aed33);
border:1px solid rgba(255,255,255,.08);
}

</style>
""",unsafe_allow_html=True)

# ---------- PREMIUM CHARTS ----------
st.markdown("## 🚀 Advanced Insights")

left,right=st.columns(2)

with left:

    sentiment_counts=df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns=["Sentiment","Count"]

    fig=px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=.65,
        title="Sentiment Split"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=430,
        margin=dict(l=20,r=20,t=60,b=20)
    )

    st.plotly_chart(fig,use_container_width=True)

with right:

    theme_counts=df["theme"].value_counts().reset_index()
    theme_counts.columns=["Theme","Count"]

    fig2=px.bar(
        theme_counts,
        x="Theme",
        y="Count",
        text="Count"
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=430,
        title="Theme Frequency"
    )

    st.plotly_chart(fig2,use_container_width=True)

st.markdown("---")

# ---------- GAUGE ----------
st.subheader("⭐ Overall Satisfaction Index")

overall=((df["rating"].mean()-1)/9)*100

gauge=go.Figure(go.Indicator(
    mode="gauge+number",
    value=overall,
    number={'suffix':"%"},
    title={'text':"Customer Satisfaction"},
    gauge={
        'axis':{'range':[0,100]},
        'bar':{'color':'#3b82f6'},
        'steps':[
            {'range':[0,40],'color':'#7f1d1d'},
            {'range':[40,70],'color':'#854d0e'},
            {'range':[70,100],'color':'#14532d'}
        ]
    }
))

gauge.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    height=420
)

st.plotly_chart(gauge,use_container_width=True)

st.markdown("---")

st.subheader("💬 Quick Insights")

positive=(df["sentiment"]=="Positive").mean()*100

if positive>80:
    st.success("Overall customer experience is highly positive.")
elif positive>60:
    st.info("Customer satisfaction is good but has room for improvement.")
else:
    st.warning("Negative feedback is significant. Investigate key service issues.")

top_theme=df["theme"].value_counts().idxmax()

st.markdown(f"""
<div class="rec">
<b>Most Mentioned Theme</b><br><br>
🏷️ {top_theme}
</div>
""",unsafe_allow_html=True)

st.markdown("""
<div class="footer-card">
<h2>✈ Flight Feedback Intelligence</h2>
<p>Built with Streamlit • Scikit-Learn • NLP • Plotly</p>
<p>Premium Dashboard UI</p>
</div>
""",unsafe_allow_html=True)


st.markdown("""
<style>

/* ---------- Smooth Fade ---------- */

@keyframes fadeUp{
0%{
opacity:0;
transform:translateY(20px);
}
100%{
opacity:1;
transform:translateY(0px);
}
}

.metric-card,
.hero,
.rec{
animation:fadeUp .7s ease;
}

/* ---------- Hover ---------- */

.metric-card{
transition:all .35s ease;
}

.metric-card:hover{
transform:translateY(-8px) scale(1.01);
box-shadow:0 18px 40px rgba(0,0,0,.45);
}

/* ---------- Buttons ---------- */

.stButton>button{
transition:.25s ease;
}

.stButton>button:hover{
box-shadow:0 0 25px rgba(59,130,246,.55);
transform:translateY(-2px);
}

/* ---------- Dataframe ---------- */

[data-testid="stDataFrame"]{
border-radius:18px;
overflow:hidden;
border:1px solid rgba(255,255,255,.08);
}

/* ---------- Tabs ---------- */

button[data-baseweb="tab"]{
font-size:18px;
font-weight:600;
padding:16px 26px;
}

button[data-baseweb="tab"][aria-selected="true"]{
border-bottom:3px solid #3b82f6;
}

/* ---------- Scrollbar ---------- */

::-webkit-scrollbar{
width:9px;
}

::-webkit-scrollbar-thumb{
background:#3b82f6;
border-radius:999px;
}

/* ---------- Selection ---------- */

::selection{
background:#3b82f6;
color:white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
"""
### 🚀 About this Project

This dashboard combines **Machine Learning**, **Natural Language Processing**
and **interactive analytics** to convert customer airline reviews into
business insights.

#### Features
- 😊 Sentiment Analysis
- ⭐ Satisfaction Prediction
- 🏷️ Theme Detection
- 💡 AI Recommendations
- 📊 Interactive Analytics Dashboard
- 🎯 Satisfaction Gauge
"""
)

with st.expander("🛠 Technology Stack"):
    st.markdown("""
- **Python**
- **Streamlit**
- **Scikit-Learn**
- **NLTK**
- **TF-IDF Vectorization**
- **Pandas & NumPy**
- **Plotly**
- **Joblib**
""")

st.markdown(
"""
<div style="
margin-top:40px;
padding:28px;
border-radius:22px;
background:rgba(255,255,255,.05);
border:1px solid rgba(255,255,255,.08);
text-align:center;
">

<h2 style="margin-bottom:8px;">✈ Flight Feedback Intelligence</h2>

<p>
Designed as a modern AI dashboard for airline customer experience analysis.
</p>

<p style="opacity:.75;">
Built using Streamlit • Machine Learning • NLP • Plotly
</p>

</div>
""",
unsafe_allow_html=True
)



