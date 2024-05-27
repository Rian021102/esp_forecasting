# Import necessary modules
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import numpy as np
import tensorflow as tf

# Define a Pydantic model for input validation
class PredictionRequest(BaseModel):
    data: conlist(conlist(float, min_items=1, max_items=1), min_items=1)  # Adjust the shape according to your input requirements

# Initialize FastAPI app
app = FastAPI()

# Load your trained model
model_path = '/path/to/your/model.h5'  # Update the path to where your model is stored
model = tf.keras.models.load_model(model_path)

@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Convert input data to a NumPy array
    input_data = np.array(request.data)
    # Reshape the data if needed, assuming the model expects data in a certain shape
    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    # Make predictions
    predictions = model.predict(input_data)
    # Optionally, process predictions here (e.g., calculate reconstruction errors)
    # Return predictions
    return {"predictions": predictions.tolist()}

# This part is for development and testing locally, not needed in production
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
