import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from clustering_utils import preprocess_data, perform_clustering, get_cluster_plot
from recommender import recommend_services

st.set_page_config(page_title="Credit Card Clustering", layout="wide")

st.title("💳 Credit Card Customer Clustering Dashboard")
st.markdown("Analyze customer segments and get personalized insights based on financial behavior.")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv('cc_general.csv')
    st.subheader("Raw Data")
    st.dataframe(data.head())

    data_clean, scaled_data = preprocess_data(data)

    clusters, model, scaled_df = perform_clustering(data_clean, scaled_data)
    data_clean["CREDIT_CARD_CLUSTERS"] = clusters
    cluster_map = {
        0: "Cluster 1",
        1: "Cluster 2",
        2: "Cluster 3",
        3: "Cluster 4",
        4: "Cluster 5"
    }
    data_clean["CREDIT_CARD_CLUSTERS"] = data_clean["CREDIT_CARD_CLUSTERS"].map(cluster_map)

    st.subheader("📊 Cluster Summary")
    st.dataframe(data_clean.groupby("CREDIT_CARD_CLUSTERS").mean(numeric_only=True))

    st.subheader("📈 3D Visualization")
    fig = get_cluster_plot(data_clean)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🤖 Predict Your Cluster")
    balance = st.number_input("Balance", value=1000.0)
    purchases = st.number_input("Purchases", value=500.0)
    credit_limit = st.number_input("Credit Limit", value=1500.0)

    if st.button("Predict Cluster"):
        user_input = pd.DataFrame([[balance, purchases, credit_limit]], columns=["BALANCE", "PURCHASES", "CREDIT_LIMIT"])
        pred_scaled = model['scaler'].transform(user_input)
        cluster = model['kmeans'].predict(pred_scaled)[0]
        cluster_label = cluster_map[cluster]
        st.success(f"Predicted Cluster: {cluster_label}")
        st.info(recommend_services(cluster_label))
