import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load Dataset
df = pd.read_csv("train.csv")

# Features
X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]

# Target
y = df["SalePrice"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "house_price_model.pkl")

print("Model Saved Successfully!")