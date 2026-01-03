from flask import Flask, request, render_template_string, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("heart_model.pkl")

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
    
    <form method="POST" action="/predict">
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
    
    {% if result %}
    <h2>Result: {{ result }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_CODE)

@app.route("/predict", methods=["POST"])
def predict():
    # Check if JSON or Form data
    if request.is_json:
        # For Postman JSON
        data = request.get_json()
        age = int(data["Age"])
        chestpain = int(data["ChestPainType"])
        bp = int(data["BP"])
        cholesterol = int(data["Cholesterol"])
        
        features = np.array([[age, chestpain, bp, cholesterol]])
        prediction = model.predict(features)
        
        result = int(prediction[0])
        
        if result == 1:
            message = "Heart Disease Present"
        else:
            message = "No Heart Disease"
        
        return jsonify({
            "prediction": result,
            "message": message
        })
    else:
        # For Browser Form
        age = int(request.form["age"])
        chestpain = int(request.form["chestpain"])
        bp = int(request.form["bp"])
        cholesterol = int(request.form["cholesterol"])
        
        features = np.array([[age, chestpain, bp, cholesterol]])
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            result = "Heart Disease Present"
        else:
            result = "No Heart Disease"
        
        return render_template_string(HTML_CODE, result=result)

if __name__ == "__main__":
    app.run(debug=True)