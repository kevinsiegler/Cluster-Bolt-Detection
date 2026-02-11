import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from config import FEATURE_PATH, MODEL_PATH, CONTAMINATION

X = np.load(FEATURE_PATH)

model = IsolationForest(
    n_estimators=300,
    contamination=CONTAMINATION,
    random_state=42
)

model.fit(X)

joblib.dump(model, MODEL_PATH)
print("Model trained and saved.")
