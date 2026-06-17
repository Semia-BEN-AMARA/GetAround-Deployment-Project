from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pydantic
import joblib
import pandas as pd
import numpy as np
import os


# Initializing API and loading the trained model

app = FastAPI(
    title="GetAround Pricing Optimization API",
    description="API de prédiction en temps réel du prix optimal de location journalier.",
    version="1.0.0"
)

# Loading the trained model from the .joblib file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pricing_model.joblib")

print(f"Chargement du modèle de production depuis : {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


FEATURES_ORDER = [
    "model_key", "fuel", "paint_color", "car_type", "mileage", "engine_power",
    "private_parking_available", "has_gps", "has_air_conditioning",
    "automatic_car", "has_getaround_connect", "has_speed_regulator", "winter_tires"
]

# defining the input schema for the /predict endpoint using Pydantic

class PredictionInput(BaseModel):

    input: list

    model_config = {
        "json_schema_extra": {
            "example": {
                "input": [
                    ["Citroën", "diesel", "black", "convertible", 140000, 100, True, True, False, False, True, True, True]
                ]
            }
        }
    }

# Endoint /predict (POST method) for real-time predictions

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Endpoint de calcul de prédiction. 
    Prend en entrée une liste de listes de caractéristiques de véhicules,
    puis renvoie le vecteur de prix optimaux calculés.
    """
    
    raw_inputs = data.input
    
    
    df_features = pd.DataFrame(raw_inputs, columns=FEATURES_ORDER)
    
    
    predictions_array = model.predict(df_features)
    
    
    predictions_clean = [float(np.round(p, 2)) for p in predictions_array]
    

    return {"prediction": predictions_clean}

# Documentation page accessible at /docs and root (/) with custom HTML content


@app.get("/", response_class=HTMLResponse)
@app.get("/docs", response_class=HTMLResponse)
async def custom_documentation():
    """
    Page de documentation principale HTML exigée par le cahier des charges du projet.
    Accessible directement à la racine (/) et sur l'endpoint (/docs).
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>GetAround API Documentation</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 40px; color: #2c3e50; background-color: #f8fafc; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            h1 { color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; margin-top: 0; }
            h2 { color: #2b6cb0; margin-top: 30px; }
            .endpoint { background: #edf2f7; padding: 20px; border-radius: 6px; border-left: 5px solid #3182ce; margin-bottom: 25px; }
            .method { font-weight: bold; padding: 4px 8px; border-radius: 4px; color: white; display: inline-block; font-size: 0.9em; }
            .post { background: #38a169; }
            .get { background: #3182ce; }
            .path { font-family: 'Courier New', Courier, monospace; font-weight: bold; font-size: 1.1em; margin-left: 10px; }
            pre { background: #1a202c; color: #aeacd4; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 0.9em; }
            .key { color: #f6ad55; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 GetAround Autogestion Pricing API</h1>
            <p>Welcome to the official GetAround Machine Learning optimization service. This microservice exposes our champion automated linear model to calculate recommendations for daily rental vehicle rates.</p>
            
            <h2>Available API Endpoints Description</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="path">/predict</span>
                <p><strong>Description:</strong> Computes the optimized suggested rental pricing per day for one or multiple cars based on technical and comfort features.</p>
                <p><strong>Required Input Payload (JSON Format):</strong></p>
                <pre>{
  <span class="key">"input"</span>: [
    [<span class="key">"Citroën"</span>, <span class="key">"diesel"</span>, <span class="key">"black"</span>, <span class="key">"convertible"</span>, 140000, 100, <span class="key">true</span>, <span class="key">true</span>, <span class="key">false</span>, <span class="key">false</span>, <span class="key">true</span>, <span class="key">true</span>, <span class="key">true</span>]
  ]
}</pre>
                <p><strong>Expected Output Response (JSON Format):</strong></p>
                <pre>{
  <span class="key">"prediction"</span>: [124.52]
}</pre>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <span class="path">/docs</span> ou <span class="path">/</span>
                <p><strong>Description:</strong> Serves this exact comprehensive custom documentation and deployment specifications guide page.</p>
                <p><strong>Expected Output:</strong> Standard HTML text document rendering.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)