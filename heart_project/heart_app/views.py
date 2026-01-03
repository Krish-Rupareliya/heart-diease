from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt 
import joblib
import numpy as np
import json

# Load model
model = joblib.load("D:\home\heart\heart_model.pkl")

def home(request):
    """Home page with form"""
    return render(request, 'predict.html')

@csrf_exempt
def predict(request):
    """Prediction function for both JSON and Form"""
    
    if request.method == "POST":
        
        # Check if JSON or Form data
        if request.content_type == 'application/json':
            # For Postman JSON
            data = json.loads(request.body)
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
            
            return JsonResponse({
                "prediction": result,
                "message": message
            })
        else:
            # For Browser Form
            age = int(request.POST.get('age'))
            chestpain = int(request.POST.get('chestpain'))
            bp = int(request.POST.get('bp'))
            cholesterol = int(request.POST.get('cholesterol'))
            
            features = np.array([[age, chestpain, bp, cholesterol]])
            prediction = model.predict(features)
            
            if prediction[0] == 1:
                result = "Heart Disease Present"
            else:
                result = "No Heart Disease"
            
            return render(request, 'predict.html', {'result': result})
    
    return render(request, 'predict.html')