from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    trip_distance: float = Field(..., gt=0)
    trip_duration_min: float = Field(..., gt=0)
    passenger_count: float = Field(..., ge=1, le=8)
    RatecodeID: float = Field(..., ge=1, le=6)
    pickup_hour: float = Field(..., ge=0, le=23)


class PredictionResponse(BaseModel):
    prediction: float
    model_uri: str
