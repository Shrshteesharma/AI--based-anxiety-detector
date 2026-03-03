import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("../data/dataset.csv")

# Handle missing values
df.dropna(inplace=True)

# Label Mapping
label_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}

df["label_num"] = df["label"].map(label_mapping)

# Features and labels
X = df["text"]
y = df["label_num"]

# Convert text into numerical vectors
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Save model and vectorizer
pickle.dump(model, open("../model/anxiety_model.pkl", "wb"))
pickle.dump(vectorizer, open("../model/vectorizer.pkl", "wb"))

print("✅ Milestone 3 completed successfully!")