import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel



model_path = os.path.join(os.path.dirname(__file__),'..','models','logistic_regression_model.joblib')
vectorizer_path = os.path.join(os.path.dirname(__file__),'..','models','tfidf_vectorizer.joblib')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

category_mapping = {0:"World",1:"Sports",2:"Business",3:"Sci/Tech"}



app = FastAPI()

class Item(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message":"News Classifier API"}

@app.post('/predict')

def predict(item: Item):
    try:
       input_data = item.text
       input_tfidf = vectorizer.transform([input_data])
       prediction = model.predict(input_tfidf)
       predicted_category = category_mapping[prediction[0]]
       return {"predicted category":predicted_category}
    except Exception as e:
        return {"error ":str(e) }
    

