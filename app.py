from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
import tensorflow as tf

# Define a Pydantic model for input validation
class PredictionRequest(BaseModel):
    data: list[list[float]]  # Expecting a list of lists with floats

    @validator('data', each_item=True)
    def check_length(cls, v):
        assert len(v) == 1, 'Each sublist must contain exactly one float'
        return v

# Initialize FastAPI app
app = FastAPI()

# Load your trained model
model_path = '/Users/rianrachmanto/miniforge3/project/esp_forecast_LSTM/autoencoder_ampere.h5'  # Update the path to where your model is stored
model = tf.keras.models.load_model(model_path)

@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Convert input data to a NumPy array
    input_data = np.array(request.data)
    # Reshape the data to match the model's expected input shape
    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    # Make predictions
    predictions = model.predict(input_data)
    # Calculate reconstruction error as an example of processing
    reconstruction_error = np.mean(np.abs(predictions - input_data), axis=1)
    # Return predictions and any other relevant information
    return {"predictions": predictions.tolist(), "reconstruction_error": reconstruction_error.tolist()}

# This part is for development and testing locally, not needed in production
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
