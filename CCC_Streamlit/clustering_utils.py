import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go

def preprocess_data(df):
    df = df.dropna()
    features = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
    t_data = df[features]
    scaler = MinMaxScaler()
    t_data_scaled = scaler.fit_transform(t_data)
    return df.copy(), t_data_scaled

def perform_clustering(df, scaled_data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    model = {
        'scaler': MinMaxScaler().fit(df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]),
        'kmeans': kmeans
    }
    return clusters, model, scaled_data

def get_cluster_plot(df):
    colors = ["red", "blue", "green", "purple", "orange"]
    fig = go.Figure()
    for idx, i in enumerate(df["CREDIT_CARD_CLUSTERS"].unique()):
        fig.add_trace(go.Scatter3d(
            x=df[df["CREDIT_CARD_CLUSTERS"] == i]['BALANCE'],
            y=df[df["CREDIT_CARD_CLUSTERS"] == i]['PURCHASES'],
            z=df[df["CREDIT_CARD_CLUSTERS"] == i]['CREDIT_LIMIT'],
            mode='markers',
            marker=dict(size=6, color=colors[idx % len(colors)], line=dict(width=1)),
            name=str(i)
        ))

    fig.update_traces(hovertemplate="BALANCE: %{x}<br>PURCHASES: %{y}<br>CREDIT_LIMIT: %{z}")
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='BALANCE'),
            yaxis=dict(title='PURCHASES'),
            zaxis=dict(title='CREDIT_LIMIT')
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig
