import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Influencer Dashboard", layout="wide")

st.title("📊 Instagram Influencer Dashboard")

# Upload CSV
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    # Read CSV
    df = pd.read_csv(file)

    # Fix column names
    df.columns = df.columns.str.title()

    # Clean Score column
    df['Score'] = df['Score'].astype(str)
    df['Score'] = df['Score'].str.replace('%', '', regex=False)
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

    # Remove NaN values
    df = df.dropna(subset=['Score'])

    # Show data
    st.subheader("📄 Data Preview")
    st.dataframe(df)

    # Sidebar controls
    st.sidebar.header("Controls")

    chart_type = st.sidebar.selectbox(
        "Select Chart Type",
        ["Bar", "Line"]
    )

    color = st.sidebar.selectbox(
        "Select Color",
        ["blue", "green", "red", "orange"]
    )

    top_n = st.sidebar.slider(
        "Select Top N Influencers",
        3, 200, 5
    )

    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        ["Linear Regression", "Random Forest"]
    )

    # Top influencers
    top_df = df.sort_values(by="Score", ascending=False).head(top_n)

    top_name = top_df.iloc[0]['Channel_Info']
    top_score = top_df.iloc[0]['Score']

    st.success(f"🏆 Top Influencer: {top_name} (Score: {top_score})")

    # Chart
    st.subheader("📈 Influencer Scores")

    fig, ax = plt.subplots(figsize=(10, 5))

    if chart_type == "Bar":
        ax.bar(top_df['Channel_Info'], top_df['Score'], color=color)
    else:
        ax.plot(top_df['Channel_Info'], top_df['Score'], color=color, marker='o')

    plt.xticks(rotation=45)
    plt.xlabel("Influencer")
    plt.ylabel("Score")

    st.pyplot(fig)

    # Prediction
    st.subheader("🤖 Prediction")

    X = np.array(range(len(top_df))).reshape(-1, 1)
    y = top_df['Score'].values

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor()

    model.fit(X, y)

    future = np.array([[len(top_df)]])
    prediction = model.predict(future)

    st.info(f"Predicted Future Score: {prediction[0]:.2f}")

else:
    st.warning("Please upload a CSV file.")
