import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Generate synthetic data
rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "area": rng.integers(500, 4000, n),
    "bedrooms": rng.integers(1, 6, n),
    "bathrooms": rng.integers(1, 4, n),
    "age": rng.integers(0, 50, n),
    "location": rng.choice(["urban", "suburban", "rural"], n),
})
loc_bonus = df["location"].map({"urban": 80000, "suburban": 40000, "rural": 0})
df["price"] = (df["area"] * 150 + df["bedrooms"] * 10000 + df["bathrooms"] * 8000
               - df["age"] * 1000 + loc_bonus + rng.normal(0, 15000, n))

X = df.drop(columns="price")
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and model combined in one pipeline
preprocess = ColumnTransformer([
    ("num", StandardScaler(), ["area", "bedrooms", "bathrooms", "age"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"]),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
print(f"R2 : {r2_score(y_test, pred):.4f}")

joblib.dump(model, "model.pkl")
print("model.pkl saved successfully")