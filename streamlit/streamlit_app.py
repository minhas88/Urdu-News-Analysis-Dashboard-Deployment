import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit.components.v1 import html


st.set_page_config(
    page_title="Urdu News Dashboard",
    page_icon="📰",
    layout="wide"
)

# DB connection
FASTAPI_URL = os.getenv("FASTAPI_URL")

CATEGORY_MAP = {
    1: "Entertainment",
    2: "Sports",
    3: "International",
    4: "Business",
    5: "Technology"
}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    try:
        response = requests.get(FASTAPI_URL, timeout=10)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Failed to fetch data from FastAPI: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"🚨 Unexpected error: {str(e)}")
        return pd.DataFrame()

df = load_data()
if not df.empty:
    df["category"] = df["predicted_label"].astype(int).map(CATEGORY_MAP)


# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/21/21601.png", width=70)
st.sidebar.header("🧭 Filter Options")

if not df.empty:
    categories = st.sidebar.multiselect("Select Categories", df["category"].unique(), default=df["category"].unique())
    sentiments = st.sidebar.multiselect("Select Sentiments", df["predicted_sentiment"].unique(), default=df["predicted_sentiment"].unique())

    filtered_df = df[
        (df["category"].isin(categories)) &
        (df["predicted_sentiment"].isin(sentiments))
    ]
    st.sidebar.success(f"🎯 Showing {len(filtered_df)} articles")
else:
    filtered_df = df
    st.sidebar.warning("No data available for filtering.")

# Live ticker
if not df.empty:
    ticker_df = df.sample(n=min(20, len(df)))
    ticker_text = " • ".join([f"{row['title']}" for _, row in ticker_df.iterrows()])

    html(f"""
        <div style="background-color:#111; padding:15px;">
            <marquee direction="left" scrollamount="3" style="color:#fff; font-size:22px; font-weight:bold;">
                🗞️ {ticker_text}
            </marquee>
        </div>
    """, height=65)

# Header
st.markdown("""
    <h1 style='text-align: center;'>📊 Urdu News Predictions Dashboard</h1>
    <p style='text-align: center;'>Explore category and sentiment analysis of Urdu news headlines using machine learning.</p>
    <hr style='border:1px solid #666;'>
""", unsafe_allow_html=True)

# Layout: Articles and Summary
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("📰 Recent News Articles")
    if not filtered_df.empty:
        sampled_df = filtered_df.sample(n=min(8, len(filtered_df))).reset_index(drop=True)
        for _, row in sampled_df.iterrows():
            st.markdown(f"""
                <div style="padding: 15px; border-radius: 12px; margin-bottom: 15px;
                            background-color: #f9f9f9; width: 95%; margin-left:auto; 
                            box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: right;">
                    <h4 style="color: black; font-size: 18px; margin-bottom: 5px;">{row['title']}</h4>
                    <p style="color: black; font-size: 14px; margin: 0;">
                        <strong>Category:</strong> {row['category']} &nbsp;&nbsp;|&nbsp;&nbsp;
                        <strong>Sentiment:</strong> {row['predicted_sentiment']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No articles found.")

with col2:
    st.subheader("📋 Summary")
    if not filtered_df.empty:
        st.metric("Total Articles", len(filtered_df))
        
        st.markdown("**Category Breakdown:**")
        cat_df = pd.DataFrame(filtered_df["category"].value_counts()).reset_index()
        cat_df.columns = ["Category", "Count"]
        st.dataframe(cat_df, hide_index=True)

        st.markdown("**Sentiment Breakdown:**")
        sent_df = pd.DataFrame(filtered_df["predicted_sentiment"].value_counts()).reset_index()
        sent_df.columns = ["Sentiment", "Count"]
        st.dataframe(sent_df, hide_index=True)

    else:
        st.info("No summary data.")

st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

# Visualizations
vis1, vis2 = st.columns(2)

with vis1:
    st.subheader("📈 Articles by Category")
    if not filtered_df.empty:
        cat_counts = filtered_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_bar = px.bar(cat_counts, x="Category", y="Count", color="Category")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No category data.")

with vis2:
    st.subheader("😊 Sentiment Distribution")
    if not filtered_df.empty:
        sent_counts = filtered_df["predicted_sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        fig_pie = px.pie(sent_counts, names="Sentiment", values="Count")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No sentiment data.")

# Sentiment by Category
st.subheader("📊 Sentiment by Category")
if not filtered_df.empty:
    grouped = filtered_df.groupby(["category", "predicted_sentiment"]).size().reset_index(name="count")
    fig_grouped = px.bar(grouped, x="category", y="count", color="predicted_sentiment", barmode="stack")
    st.plotly_chart(fig_grouped, use_container_width=True)
else:
    st.info("No sentiment-category data available.")
