from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Load the trained MobileNet model
MODEL_PATH = "my_model.keras"  # Updated to .keras format

try:
    model = load_model(MODEL_PATH, compile=False)  # Load model without compiling
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Define class labels (modify according to your dataset)
CLASS_LABELS = ["Normal", "Pneumonia"]

def preprocess_image(file) -> np.ndarray:
    """Preprocesses an image for model prediction."""
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))  # Resize to MobileNet input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Receives an image, preprocesses it, and returns a model prediction."""
    try:
        img_array = preprocess_image(file.file)
        predictions = model.predict(img_array)
        class_index = int(np.argmax(predictions))  # Get predicted class index
        confidence = float(np.max(predictions))  # Get confidence score
        return {"class": CLASS_LABELS[class_index], "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "FastAPI MobileNet Inference Server is Running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
