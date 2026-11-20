
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np

# --------------------------------------------------
# Load the saved model artifact
# --------------------------------------------------

artifact = joblib.load("RoadSafeAI_Best_Model.pkl")

model = artifact["model"]
FEATURES = artifact["features"]
FATAL_MULTIPLIER = artifact["fatal_multiplier"]

app = FastAPI(
    title="RoadSafe AI - Accident Severity Prediction API",
    description="Predicts UK road accident severity using a trained ML model.",
    version="1.0.0"
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class AccidentInput(BaseModel):

    Weather_Conditions: str
    Light_Conditions: str
    Road_Surface_Conditions: str
    Road_Type: str

    Speed_limit: str
    Urban_or_Rural_Area: str

    first_road_class: str = Field(alias="1st_Road_Class")
    Junction_Detail: str
    Junction_Control: str

    Number_of_Vehicles: float

    Special_Conditions_at_Site: str

    second_road_class: str = Field(alias="2nd_Road_Class")
    second_road_number: str = Field(alias="2nd_Road_Number")
    first_road_number: str = Field(alias="1st_Road_Number")

    Hour: str
    Weekday: str
    Month: str

    class Config:
        populate_by_name = True


# --------------------------------------------------
# Feature preprocessing
# --------------------------------------------------

def preprocess_input(data: AccidentInput):

    row = data.model_dump(by_alias=True)

    # Convert categorical features to strings
    for column in FEATURES:

        if column != "Number_of_Vehicles":
            row[column] = str(row[column]).strip()

            if row[column] == "":
                row[column] = "<blank>"

    # Road number feature engineering
    for prefix in ["1st", "2nd"]:

        number_col = f"{prefix}_Road_Number"
        class_col = f"{prefix}_Road_Class"

        number = row[number_col]

        try:
            is_positive_number = float(number) > 0
        except (ValueError, TypeError):
            is_positive_number = False

        if is_positive_number:
            row[number_col] = f"{row[class_col]}|{number}"

    # Match notebook Junction_Control cleaning logic
    off_junction = "Not at junction or within 20 metres"
    missing_code = "Data missing or out of range"

    if (
        row["Junction_Detail"] == off_junction
        and row["Junction_Control"] == missing_code
    ):
        row["Junction_Control"] = "Not applicable - off junction"

    if row["Junction_Control"] == off_junction:
        row["Junction_Control"] = "Not applicable - off junction"

    # Number of vehicles is numeric
    row["Number_of_Vehicles"] = float(row["Number_of_Vehicles"])

    # Create DataFrame with the exact feature order
    input_df = pd.DataFrame([row])

    input_df = input_df[FEATURES]

    return input_df


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to RoadSafe AI",
        "model": artifact["model_name"],
        "status": "API is running",
        "documentation": "/docs"
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_accident_severity(data: AccidentInput):

    try:

        input_df = preprocess_input(data)

        # Get class probabilities
        probabilities = model.predict_proba(input_df)

        classes = model.classes_

        # Apply saved Fatal multiplier
        fatal_index = list(classes).index("Fatal")

        adjusted_probabilities = probabilities.copy()

        adjusted_probabilities[:, fatal_index] *= FATAL_MULTIPLIER

        # Final prediction
        prediction_index = adjusted_probabilities.argmax(axis=1)

        predicted_severity = classes[prediction_index][0]

        # Probability dictionary
        probability_dict = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(classes, adjusted_probabilities[0])
        }

        return {
            "predicted_severity": str(predicted_severity),
            "probabilities": probability_dict,
            "fatal_multiplier": FATAL_MULTIPLIER,
            "model_name": artifact["model_name"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
