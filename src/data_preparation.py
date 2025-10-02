import numpy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

train_data = pandas.read_csv('data/train.csv')
test_data = pandas.read_csv('data/test.csv')

print(train_data.head())
print(test_data.head())

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2),max_features=5000)

x_train = vectorizer.fit_transform(train_data["text"])
x_test = vectorizer.transform(test_data["text"])

print(x_train.shape)
print(x_test.shape)

train_data['label'].value_counts().plot(kind='bar', color='orange')
plt.title("Class distribution in training set")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(ticks=range(4),labels=["World","Sports","Business","Sci/Tech"],rotation=0)
plt.show()

text = " ".join(train_data["text"].values)

wordcloud = WordCloud(stopwords="english",background_color="white").generate(text)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()