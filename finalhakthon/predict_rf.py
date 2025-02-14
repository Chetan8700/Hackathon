import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv("C:/Users/CHETAN/Downloads/AP Primary Sector Farmerwise Soil Health Data of 13 districts.csv")
data.head()

# Remove unnecessary columns
keep = ["pH", "EC", "OC", "Avail-P", "Exch-K", "Avail-S", 
        "Avail-B", "Avail-Zn", "Avail-Fe", "Avail-Cu", "Avail-Mn"]

data = data.drop(columns=[col for col in data.columns if col not in keep])
data.to_csv("cleaned_data.csv", index=False)
data.head()

# Handle missing values
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.mean(), inplace=True)

def classify_soil(row):
    if (
        5.5 <= row["pH"] <= 8.0 and
        row["EC"] < 3 and
        row["OC"] >= 0.3 and
        row["Avail-P"] >= 8 and
        row["Exch-K"] >= 80 and
        row["Avail-S"] >= 8 and
        row["Avail-B"] >= 0.4 and
        row["Avail-Zn"] >= 0.5 and
        row["Avail-Fe"] >= 3.5 and
        row["Avail-Cu"] >= 0.15 and
        row["Avail-Mn"] >= 1.5
    ):
        return 1
    else:
        return 0

data["health"] = data.apply(classify_soil, axis=1)
data.to_csv("updated_health.csv", index=False)
data.head()

# Prepare data for training
X = data.drop(columns=["health"])
y = data["health"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model as a joblib file
joblib.dump(rf_model, "rf_model.joblib")
print("Random Forest model saved as 'rf_model.joblib'.")

# Evaluate the Random Forest model
y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Personalized input prediction
def get_user_input():
    print("Enter the values for the following soil parameters:")
    pH = float(input("pH: "))
    EC = float(input("EC: "))
    OC = float(input("OC: "))
    Avail_P = float(input("Avail-P: "))
    Exch_K = float(input("Exch-K: "))
    Avail_S = float(input("Avail-S: "))
    Avail_Zn = float(input("Avail-Zn: "))
    Avail_B = float(input("Avail-B: "))
    Avail_Fe = float(input("Avail-Fe: "))
    Avail_Cu = float(input("Avail-Cu: "))
    Avail_Mn = float(input("Avail-Mn: "))

    user_input = np.array([[pH, EC, OC, Avail_P, Exch_K, Avail_S, Avail_Zn, Avail_B, Avail_Fe, Avail_Cu, Avail_Mn]])
    return user_input

def interpret_prediction(prediction):
    return "Healthy" if prediction == 1 else "Unhealthy"

user_data = get_user_input()
rf_prediction = rf_model.predict(user_data)
rf_result = interpret_prediction(rf_prediction[0])

print(f"\ud83c\udf31 Soil Health Prediction: {rf_result}")
