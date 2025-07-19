def recommend_services(cluster_label):
    recommendations = {
        "Cluster 1": "🔐 You qualify for Premium Travel Credit Cards with high rewards.",
        "Cluster 2": "💸 Consider EMI-based offers for purchases over ₹10,000.",
        "Cluster 3": "📈 Eligible for an increase in credit limit. Explore now!",
        "Cluster 4": "🛍️ Spend is low. Explore shopping offers with 0% interest EMIs.",
        "Cluster 5": "📉 Risk of churn. Offer retention incentives like reward multipliers."
    }
    return recommendations.get(cluster_label, "⚠️ No specific recommendation available.")
