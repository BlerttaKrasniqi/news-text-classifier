import joblib
import numpy

model = joblib.load("models/logistic_regression_model.joblib")

vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

category_mapping = {0:"World",1:"Sports",2:"Business",3:"Sci/Tech"}

def predict_category(sentence):
    sentence_tfidf = vectorizer.transform([sentence])
    prediction = model.predict(sentence_tfidf)
    predicted_category = category_mapping[prediction[0]]
    return predicted_category


if __name__ == "__main__":
    print("Enter a news article, or type exit to quit")

    while True:
        user_input = input("Write: ")
        if user_input.lower() == "exit":
            break

        category = predict_category(user_input)
        print(f"Predicted category: {category}")