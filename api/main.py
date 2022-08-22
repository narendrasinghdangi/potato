from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app=FastAPI()

MODEL = tf.keras.models.load_model("C:/Users/narendra/Deep Learn/Deep Learning/potato disease/saved_models/1")
CLASS_NAME= ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I think you are alive"

def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file:UploadFile = File(...)
):
     image=read_file_as_image( await file.read())
     img_batch=np.expand_dims(image,0)
     predictions= MODEL.predict(img_batch)
     class_name= CLASS_NAME[np.argmax(predictions[0])]
     confidence=np.max(predictions[0])
     return{
        "class name":class_name,
        "confidence":float(confidence)
     }
     

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)