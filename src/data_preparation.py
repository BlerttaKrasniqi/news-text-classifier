import numpy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pandas.read_csv('data/train.csv')
test_data = pandas.read_csv('data/test.csv')

print(train_data.head())
print(test_data.head())

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2),max_features=5000)

x_train = vectorizer.fit_transform(train_data["text"])
x_test = vectorizer.transform(test_data["text"])

print(x_train.shape)
print(x_test.shape)