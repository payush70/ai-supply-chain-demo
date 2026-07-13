import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load your dataset (replace with your actual CSV path)
# Features: e.g., lead_time, order_volume, weather_index, supplier_reliability
# Target: 1 (Disrupted) or 0 (On-Time)
data = pd.read_csv("supply_chain_data.csv")

# 2. Define features (X) and target label (y)
X = data.drop(columns=['disruption_status'])
y = data['disruption_status']

# 3. Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest model
# n_estimators=100 means the forest will consist of 100 decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)

# 5. Evaluate the model (Use these printouts for your dissertation's Results chapter)
predictions = rf_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 6. Export the trained model so Streamlit can use it without retraining
joblib.dump(rf_model, 'rf_predictive_model.pkl')
print("Model successfully saved as rf_predictive_model.pkl")