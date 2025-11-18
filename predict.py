import pickle
import numpy as np

# Load trained model
with open("model_rf.bin", "rb") as f:
    model = pickle.load(f)

# Example input: 30 numerical features (replace with real values when needed)
sample_input = np.array([[
    14.5, 20.4, 96.7, 600.0, 0.12, 0.09, 0.07, 0.05, 0.18, 0.07,
    0.35, 1.5, 2.4, 30.0, 0.006, 0.02, 0.03, 0.01, 0.02, 0.004,
    16.2, 28.0, 110.0, 800.0, 0.14, 0.20, 0.23, 0.11, 0.28, 0.08
]])

# Predict probability for malignant class (1)
prediction = model.predict_proba(sample_input)[0, 1]

print(f"Predicted probability (malignant): {prediction:.4f}")