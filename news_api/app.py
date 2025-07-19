import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart News Recommender", page_icon="🗞️", layout="wide")

st.title("🧠 Smart News Recommender")
st.write("Enter a news topic or query, and get the most relevant real-world news articles.")

query = st.text_input("🔍 Enter your query (e.g., Biden campaign, AI in healthcare):")

API_KEY = "c7cc75bd77524829a4e697091e7d2f1d"  # 👉 Replace with your own key from https://newsapi.org/
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_news_articles(query_term):
    params = {
        'q': query_term if query_term else "general",
        'language': 'en',
        'pageSize': 10,
        'apiKey': API_KEY
    }
    response = requests.get(NEWS_ENDPOINT, params=params)
    data = response.json()
    articles = data.get("articles", [])
    return articles

def recommend_articles(query, articles):
    if not query:
        return [], []

    combined_texts = [article['title'] + " " + (article['description'] or "") for article in articles]
    combined_texts.append(query)  # Append user query as the last item

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Compare query (last row) with all others
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    return similarity_scores, articles

if query:
    with st.spinner("Fetching relevant news..."):
        articles = fetch_news_articles(query)
        if not articles:
            st.error("No news articles found.")
        else:
            scores, ranked_articles = recommend_articles(query, articles)
            sorted_results = sorted(zip(ranked_articles, scores), key=lambda x: x[1], reverse=True)

            st.subheader("📌 Top Recommendations:")
            for idx, (article, score) in enumerate(sorted_results[:5], start=1):
                st.markdown(f"### {idx}. [{article['title']}]({article['url']})")
                st.write(f"📰 **{article['source']['name']}** | 📅 *{article['publishedAt'][:10]}*")
                st.write(article['description'] or "No description available.")
                st.write(f"📊 **Relevance Score:** {round(score * 100, 2)}")
                st.markdown("---")