import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Loading data
df = pd.read_csv('data/housing.csv')
df = df.sample(n=5000, random_state=42)  # Optional: reduce data size
df = df.dropna()

# Features and Target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# One-hot encode categorical columns (e.g., ocean_proximity)
X = pd.get_dummies(X, drop_first=True)

# Remove infinities if any
X.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs('saved_model', exist_ok=True)
joblib.dump(model, 'saved_model/model.pkl')

print(" Model trained and saved successfully.")
