import csv, os
import tensorflow_datasets as tfds

os.makedirs("data",exist_ok=True)


def write(dataset,path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text","label"])
        for ex in tfds.as_numpy(dataset):
            title = ex["title"].decode("utf-8")
            description = ex["description"].decode("utf-8")
            text = (title + ". "+description).strip()
            w.writerow([text, int(ex["label"])])

(dataset_train, dataset_test), _ = tfds.load(
    "ag_news_subset", split=["train","test"],with_info=True,as_supervised=False
)

write(dataset_train, "data/train.csv")
write(dataset_test,"data/test.csv")
print("Wrote data/train.csv and data/test.csv") 