
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "fake_news_detection_1000 - fake_news_detection_1000.csv"

FEATURES = [
    "source_credibility",
    "author_credibility",
    "clickbait_score",
    "sentiment_score",
    "article_length",
    "num_quotes",
    "num_images",
    "num_links",
    "social_shares",
    "verified_source"
]

# Load dataset
df = pd.read_csv(DATASET_PATH)

X = df[FEATURES]
y = df["label"].astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build and train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(
    y_test,
    predictions,
    target_names=["Real", "Fake"],
    zero_division=0
))

# Save trained model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")
