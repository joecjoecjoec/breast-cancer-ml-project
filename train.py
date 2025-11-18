import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_csv("breast-cancer.csv")

# 2. Basic cleaning and target encoding
df.columns = (
    df.columns
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

df["target"] = df["diagnosis"].map({"M": 1, "B": 0})
df = df.drop(columns=["id", "diagnosis"], errors="ignore")

X = df.drop(columns=["target"])
y = df["target"]

# 3. Train final model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
model.fit(X, y)

# 4. Save model
with open("model_rf.bin", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to model_rf.bin")