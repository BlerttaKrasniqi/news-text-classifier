import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
import joblib

train_data = pandas.read_csv("data/train.csv")
test_data = pandas.read_csv("data/test.csv")

vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

x_train = vectorizer.transform(train_data["text"])
x_test = vectorizer.transform(test_data["text"])

y_train = train_data["label"]
y_test = test_data["label"]

model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

y_prediction = model.predict(x_test)

accuracy = accuracy_score(y_test,y_prediction)
print(f"Accuracy score: {accuracy}")
print(classification_report(y_test,y_prediction))

joblib.dump(model,'models/logistic_regression_model.joblib')
print("Model saved as models/logistic_regression_model.joblib")