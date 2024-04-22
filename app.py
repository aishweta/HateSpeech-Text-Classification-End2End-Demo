# Hosts the REST API
from fastapi import FastAPI
from pydantic import BaseModel
from predict_model import predict

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict/")
def get_prediction(input_data: InputData):
    pretrained_model_path = './models/best_model.pth'
    prediction = predict(input_data.text, pretrained_model_path)
    return {"prediction": prediction}
