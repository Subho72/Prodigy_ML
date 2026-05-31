import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Save Model
joblib.dump(kmeans, "kmeans_model.pkl")

print("Model Saved Successfully!")