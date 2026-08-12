import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# Ensure 'Bank Customer Churn Prediction.csv' is placed in the same project directory
df = pd.read_csv("Bank Customer Churn Prediction.csv")

# Step 2: Drop non-informative identification columns
df = df.drop(columns=["customer_id"])

# Step 3: Categorical Feature Encoding
# Convert binary gender feature to numerical (Female: 0, Male: 1)
df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

# Perform One-Hot Encoding for multi-category feature 'country'
# This creates dummy variables: country_Germany and country_Spain (France is baseline)
df = pd.get_dummies(df, columns=["country"], drop_first=True)

# Step 4: Separate input features (X) and target output (y)
X = df.drop(columns=["churn"])
y = df["churn"]

# Step 5: Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Feature Scaling
# Logistic Regression requires scaled inputs due to distance-based optimization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 7: Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 8: Save the trained model and feature scaler to disk
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(
    "✅ Model and Scaler successfully trained and saved as 'churn_model.pkl'"
    " and 'scaler.pkl'!"
)