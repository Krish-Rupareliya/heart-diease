from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("heart_model.pkl")

# For JSON input (Postman)
class PredictRequest(BaseModel):
    Age: int
    ChestPainType: int
    BP: int
    Cholesterol: int

# HTML code inside Python
HTML_CODE = """
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Prediction</title>
</head>
<body>
    <h1>Heart Disease Prediction</h1>
    <h3>Enter Patient Details</h3>
    
    <form method="POST" action="/predict_form">
        <p>
            <label>Age:</label><br>
            <input type="number" name="age" required>
        </p>
        
        <p>
            <label>Chest Pain Type:</label><br>
            <select name="chestpain" required>
                <option value="">Select</option>
                <option value="1">Type 1</option>
                <option value="2">Type 2</option>
                <option value="3">Type 3</option>
                <option value="4">Type 4</option>
            </select>
        </p>
        
        <p>
            <label>Blood Pressure:</label><br>
            <input type="number" name="bp" required>
        </p>
        
        <p>
            <label>Cholesterol:</label><br>
            <input type="number" name="cholesterol" required>
        </p>
        
        <p>
            <button type="submit">Predict</button>
        </p>
    </form>
    
    {result_html}
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_CODE.format(result_html="")

# For Postman - JSON input
@app.post("/predict")
def predict_json(data: PredictRequest):
    features = np.array([[data.Age, data.ChestPainType, data.BP, data.Cholesterol]])
    prediction = model.predict(features)
    
    result = int(prediction[0])
    
    if result == 1:
        message = "Heart Disease Present"
    else:
        message = "No Heart Disease"
    
    return {
        "prediction": result,
        "message": message
    }

# For Browser - Form input
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(age: int = Form(), chestpain: int = Form(), bp: int = Form(), cholesterol: int = Form()):
    
    features = np.array([[age, chestpain, bp, cholesterol]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        result = "Heart Disease Present"
    else:
        result = "No Heart Disease"
    
    result_html = f"<h2>Result: {result}</h2>"
    return HTML_CODE.format(result_html=result_html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)