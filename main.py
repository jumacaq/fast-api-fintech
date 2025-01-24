from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from contextlib import asynccontextmanager

# Define the input data structure
class FormData(BaseModel):
    knowledgeLevel: str
    goals: list[str]
    riskPreference: str
    monthlyIncome: float
    monthlyExpenses: float
    savingsPercentage: float

# Define the recommendation function (from Google Colab)
def recommend_assets(formData, model, mlb, encoder, label_encoder, tickers, top_n=3):
    # Prepare input data
    input_data = pd.DataFrame([{
        'knowledgeLevel': formData['knowledgeLevel'],
        'goals': formData['goals'],
        'riskPreference': formData['riskPreference'],
        'monthlyIncome': formData['monthlyIncome'],
        'monthlyExpenses': formData['monthlyExpenses'],
        'savingsPercentage': formData['savingsPercentage'],
    }])

    # Calculate financial_stability
    input_data['financial_stability'] = (
        (input_data['monthlyIncome'] - input_data['monthlyExpenses']) / input_data['monthlyIncome']
    )

    # Encode `goals` using MultiLabelBinarizer
    goals_encoded = pd.DataFrame(
        mlb.transform(input_data['goals']),
        columns=mlb.classes_,
        index=input_data.index
    )

    # Encode categorical features using OneHotEncoder
    encoded_features = pd.DataFrame(
        encoder.transform(input_data[['knowledgeLevel', 'riskPreference']]),
        columns=encoder.get_feature_names_out(['knowledgeLevel', 'riskPreference']),
        index=input_data.index
    )

    # Combine all features
    input_data = pd.concat([input_data, encoded_features, goals_encoded], axis=1)
    input_data = input_data.drop(columns=['knowledgeLevel', 'riskPreference', 'goals'])

    # Ensure all tickers are present (dummy encoding)
    for ticker in tickers:
        input_data[f'ticker_{ticker}'] = 0

    # Align columns with training data
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict probabilities using the trained model
    probabilities = model.predict_proba(input_data)

    # Ensure probabilities shape is correct
    print(f"Probabilities shape: {probabilities.shape}")

    # Extract top N recommendations
    top_n_indices = np.argsort(probabilities[0])[-top_n:][::-1]

    # Transform top indices into asset names
    recommended_assets = label_encoder.inverse_transform(top_n_indices)

    # Return the top N assets
    return recommended_assets.tolist()

# Initialize FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, mlb, encoder, label_encoder, tickers
    model = joblib.load("model.pkl")  # Load trained model
    mlb = joblib.load("mlb.pkl")  # Load MultiLabelBinarizer
    encoder = joblib.load("encoder.pkl")  # Load OneHotEncoder
    label_encoder = joblib.load("label_encoder.pkl")  # Load LabelEncoder
    tickers = joblib.load("tickers.pkl")   # Example tickers (adjust as needed)
    yield  # Hand over control to the app
    # Cleanup (if needed)
    del model, mlb, encoder, label_encoder, tickers

app = FastAPI(lifespan=lifespan)

# Define the recommendation endpoint
@app.post("/recommend-assets/")
def recommend_assets_endpoint(formData: FormData):
    try:
        # Convert FormData to dictionary
        form_dict = formData.model_dump()

        # Call the recommendation function
        recommended_assets = recommend_assets(
            formData=form_dict,
            model=model,
            mlb=mlb,
            encoder=encoder,
            label_encoder=label_encoder,
            tickers=tickers,
            top_n=3
        )

        return {"recommended_assets": recommended_assets}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
