from fastapi import FastAPI
from pydantic import BaseModel
from scripts.predict import predict_summary

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "T5 özetleme servisi aktif."}

@app.post("/predict/")
def predict(input: InputText):
    summary = predict_summary(input.text)
    return {"summary": summary}


# python -m uvicorn main:app --reload