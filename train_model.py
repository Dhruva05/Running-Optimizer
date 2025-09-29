# train_model.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- 1. Load and preprocess data ---
df = pd.read_csv("activities.csv")
df = df[df["Activity Type"] == "Run"].copy()
df["duration_min"] = df["Elapsed Time"] / 60

# Keep only ~5K runs
df_5k = df[(df["Distance"] >= 4.5) & (df["Distance"] <= 5.5)].copy()

# Features
features = ["Average Speed", "Elevation Gain", "Weather Temperature"]
df_5k = df_5k.dropna(subset=features + ["duration_min"])

X = df_5k[features]
y = df_5k["duration_min"]

# --- 2. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Evaluate ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Model trained. MAE: {mae:.2f} minutes")

# --- 5. Save model ---
joblib.dump(model, "race_predictor_5k.pkl")
print("💾 Model saved as race_predictor_5k.pkl")
