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


# vectorizer = TfidfVectorizer(stop_words="english",max_features=10)

features_names = vectorizer.get_feature_names_out()

x_train_dense = x_train.toarray()

for i,label in enumerate(["World","Sports","Business","Sci/Tech"]):
    print(f"Top 10 words for {label} class:")
    top_indicies = x_train_dense[train_data['label']==i].mean(axis=0).argsort()[::-1]
    top_words = [features_names[idx] for idx in top_indicies[:10]]
    print(", ".join(top_words))


    print(f"Missing values in train set: {train_data.isnull().sum()}")
    print(f"Missing values in test set: {test_data.isnull().sum()}")

    print(f"Duplicate datas in train set: {train_data.duplicated().sum()}")
    print(f"Duplicate datas in test set: {test_data.duplicated().sum()}")