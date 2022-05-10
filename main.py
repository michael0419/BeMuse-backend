from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64

from tensorflow import expand_dims
from tensorflow import dtypes

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling

from tensorflow.io import decode_image
from tensorflow.image import rgb_to_grayscale
from tensorflow.nn import softmax

from numpy import argmax
from numpy import max
from numpy import array
from json import dumps

import io

from uvicorn import run
import os

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

#Load Model

model_dir = "simpleFER.h5"
model = load_model(model_dir)

class_predictions = array([
    "mad", "happy", "neutral", "sad",
])

#Define items for incoming json

class base64Img(BaseModel):
    img64: str 

#Handle Requests

@app.get("/")
async def root():
    return {"message": "Welcome to the Simple FER API!"}

@app.post("/net/image/prediction/")
async def get_net_image_prediction(imageEncoded: base64Img = None):
    
    if imageEncoded == None:
        return {"message": "No image provided"}
    
    #image =  base64.b64decode(imageEncoded.img64)
    image = base64.b64decode(str(imageEncoded.img64))
    #img = image
    #image_stream = io.StringIO(image)

    tensor = decode_image(image, channels=3, dtype= 'float32')
    tensor = expand_dims(tensor, 0)
    
    tensor = rgb_to_grayscale(tensor)
    
    resize_and_rescale = Sequential([
      #layers.Resizing(IMG_SIZE, IMG_SIZE),
      Rescaling(1./255)
    ])
    
    tensor = resize_and_rescale(tensor)
    

    pred = model.predict(tensor)
    score = softmax(pred[0])

    class_prediction = class_predictions[argmax(score)]
    print(class_prediction)
    model_score = round(max(score) * 100, 2)
    print(model_score)

    return {
        "model-prediction": class_prediction,
        "model-prediction-confidence-score": model_score
    }

    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)