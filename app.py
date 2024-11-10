# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import load_model, preprocess_image

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model
model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()
        image = Image.open(io.BytesIO(image))
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
        
        # Determine class and confidence
        prediction_class = "cataract" if prediction > 0.5 else "no cataract"
        confidence = float(prediction) if prediction_class == "cataract" else 1 - float(prediction)
        
        return JSONResponse(content={"prediction": prediction_class, "confidence": confidence})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
