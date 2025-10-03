import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from sklearn.naive_bayes import MultinomialNB
import seaborn
import matplotlib.pyplot as plt

train_data = pandas.read_csv("data/train.csv")
test_data = pandas.read_csv("data/test.csv")

vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

x_train = vectorizer.transform(train_data["text"])
x_test = vectorizer.transform(test_data["text"])

y_train = train_data["label"]
y_test = test_data["label"]

model = LogisticRegression(max_iter=200)
nb_model = MultinomialNB()

model.fit(x_train,y_train)
y_prediction = model.predict(x_test)

nb_model.fit(x_train,y_train)
y_prediction_nb = nb_model.predict(x_test)

print("Logistic Regression Model: ")
accuracy = accuracy_score(y_test,y_prediction)
print(f"Accuracy score: {accuracy}")
print(classification_report(y_test,y_prediction))


print("Naive Bayes Model: ")

accuracy_nb = accuracy_score(y_test,y_prediction_nb)
print(f"Accuracy score: {accuracy_nb}")
print(classification_report(y_test,y_prediction_nb))

confusionmatrix_logreg = confusion_matrix(y_test, y_prediction)

confusionmatrix_nb = confusion_matrix(y_test, y_prediction_nb)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

seaborn.heatmap(confusionmatrix_logreg, annot=True, fmt='d', cmap='Blues', xticklabels=["World", "Sports", "Business", "Sci/Tech"], yticklabels=["World", "Sports", "Business", "Sci/Tech"], ax=ax[0])
ax[0].set_title('Logistic Regression Confusion Matrix')

seaborn.heatmap(confusionmatrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=["World", "Sports", "Business", "Sci/Tech"], yticklabels=["World", "Sports", "Business", "Sci/Tech"], ax=ax[1])
ax[1].set_title('Naive Bayes Confusion Matrix')

plt.tight_layout()
plt.show()


joblib.dump(model,'models/logistic_regression_model.joblib')
print("Model saved as models/logistic_regression_model.joblib")