import urllib.parse
import tensorflow_datasets as tfds

(dataset_train, dataset_test), info = tfds.load(
    "ag_news_subset", split=["train","test"], with_info=True, as_supervised=False
)

print(info)
print("Dataset is downloaded")