import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# load csv
df = pd.read_csv("D:\home\heart\Heart_Disease_Prediction.csv")

# Only 4 features
df_selected = df[['Age', 'Chest pain type', 'BP', 'Cholesterol']]

# Rename columns
df_selected.columns = ['Age', 'ChestPainType', 'BP', 'Cholesterol']

X = df_selected

# Presence=1, Absence=0
y = df["Heart Disease"].map({'Presence': 1, 'Absence': 0})

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# save model
joblib.dump(model, "heart_model.pkl")

print("Model trained and saved as heart_model.pkl")
print(f"Features used: {X.shape[1]}")
print(f"Accuracy: {model.score(X_test, y_test):.2f}")