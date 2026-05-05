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
    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df)

    # Sort data
    df = df.sort_values(by="Score", ascending=False)

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")

    chart_type = st.sidebar.selectbox(
        "Select Chart Type", ["Bar", "Line", "Pie", "Scatter"]
    )

    color = st.sidebar.selectbox(
        "Select Color", ["blue", "green", "orange", "purple", "red"]
    )

    top_n = st.sidebar.slider("Select Top N Influencers", 3, len(df), 5)

    model_type = st.sidebar.selectbox(
        "Select Prediction Model", ["Linear Regression", "Random Forest"]
    )

    df = df.head(top_n)

    # Chart
    st.subheader("📊 Visualization")

    fig, ax = plt.subplots(figsize=(4,4))   # small

    if chart_type == "Bar":
        ax.bar(df['Channel_info'], df['Score'], color=color)

    elif chart_type == "Line":
        ax.plot(df['Channel_info'], df['Score'], marker='o', color=color)

    elif chart_type == "Pie":
        ax.pie(df['Score'], labels=df['Channel_info'], autopct='%1.1f%%')

    elif chart_type == "Scatter":
        ax.scatter(df['Channel_info'], df['Score'], color=color)

    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Top influencer
    top = df.iloc[0]
    st.success(f"🏆 Top Influencer: {top['Channel_info']} (Score: {top['Score']})")

    # Prediction
    st.subheader("🤖 Prediction")

    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Score'].values

    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor()

    model.fit(X, y)
    pred = model.predict([[len(df)]])

    st.info(f"📌 Predicted Next Score: {pred[0]:.4f}")

    # Trend line
    st.subheader("📈 Trend Analysis")

    fig2, ax2 = plt.subplots()
    ax2.plot(y, label="Actual")
    ax2.plot(model.predict(X), linestyle='--', label="Trend")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.warning("Please upload your CSV file to start.")