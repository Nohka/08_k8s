from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException

from app.model_loader import load_model_from_mlflow
from app.schemas import PredictionRequest, PredictionResponse

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NYC Green Taxi Fare Prediction API")

templates = Jinja2Templates(directory="app/templates")  # HTML
app.mount("/static", StaticFiles(directory="app/static"), name="static")  # CSS

model = None
model_uri = None
model_error = None  # added in case we start the app and no experiment name exists yet, because I haven't trained a model


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Get the UI
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_uri": model_uri,
            "model_loaded": model is not None,
            "model_error": model_error,
        },
    )


@app.on_event("startup")
def startup_event():
    global model, model_uri, model_error
    try:
        model, model_uri = load_model_from_mlflow()
        model_error = None
        print(f"Startup loaded model URI: {model_uri}")
    except Exception as e:
        model = None
        model_uri = None
        model_error = str(e)
        print(f"Model loading skipped: {model_error}")


@app.get("/health")
def health():
    # Kubernetes will use this endpoint to check whether the app is alive
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_error,
    }


@app.get("/model-info")
def model_info():
    # Which model? Correct run?
    return {
        "model_uri": model_uri,
    }


@app.post("/reload-model")
def reload_model():
    global model, model_uri, model_error
    try:
        model, model_uri = load_model_from_mlflow()
        model_error = None
        return {"status": "ok", "model_uri": model_uri}
    except Exception as e:
        model_error = str(e)
        return {"status": "error", "error": model_error}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Do the inference!
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    input_df = pd.DataFrame(
        [
            {
                "trip_distance": request.trip_distance,
                "trip_duration_min": request.trip_duration_min,
                "passenger_count": request.passenger_count,
                "RatecodeID": request.RatecodeID,
                "pickup_hour": request.pickup_hour,
            }
        ]
    )

    prediction = model.predict(input_df)[0]

    return PredictionResponse(
        prediction=float(prediction),
        model_uri=model_uri,
    )
