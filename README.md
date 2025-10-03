# Dataset: AG News Subset
- The AG News Subset dataset is used for text classification task with 4 categories of news articles: World, Business, Sports, and Sci/Tech.
- The dataset was downloaded and cached using **TensorFlow Datasets (TFDS)**. Additionally, the dataset was exported to CSV format for easier
manipulation and model training.
- Dataset link: https://www.tensorflow.org/datasets/catalog/ag_news_subset
## Data loading and Export Process

## Exporting to CSV
foto

## Data preprocessing and cleaning
- In this phase, several data preparation and cleaning steps were performed, to ensure that the data is in a suitable format for training the machine learning model.
- Steps taken:
- 1. **Loading the Dataset**:
     - We load the training and testing datasets from CSV files using **pandas.read_csv()**.
     - The dataset consists of news articles with their corresponding labels (World, Sports, Business, Sci/Tech).
       fotoo
- 2. **Text Vectorization with TF-IDF**:
      - TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is used from **scikit-learn** to transform the text data into numerical features.
      - The **stop-words="english"** parameter removes common English stopwords (the, and, is)/
      - Bigrams **(ngram_range=(1,2))** is used to capture word pairs.
        fotooja e transofrmit
      - Transformation of training data and testing data using the same vectorizer
     foto
        - Trained vectorizer is saved using **joblib** so it can be reused for prediction.
          fotooo
  - 3. **Class Distribution Visualization**
       - Distribution of the news categories in the training set was visualized using a bar chart.
       - The chart shows how many samples belong to each category (World, Sports, Business, Sci/Tech).
         fotoja e kodit
  - 4. **World Cloud Visualization**
       - World Cloud is generated to visualize the most frequent words in the training set.
       - This helps understanding the key terms associated with the news articles and their categories.
      
       fotooja e kodit
  - 5. **Top words for each category**
       - We compute the top 10 words for each news category using the TF-IDF values.
       - For each category (World, Sports, Business, Sci/Tech), we compute the mean TF-IDF score for each word, sort them by importance, and print the top 10 words.
      fotojajaa
  - 6. ** Missing Values and Duplicates*
       - Missing values were checked both in training set and in testing set. This ensures that no data is missing in the text or label columns, which is important for accurate model training.
      fotoooo


## Model training
- In this phase, the machine learning models were trained for the task of classifying news articles into four categories: World, Sports, Business, and Sci/tech. The goal of this phase is to build a classification model, evaluate its performance, and compare models to identify the best one.
- 1. **Model Selection**
     - Logistic Regression is used as the primary model for text classification. Logistic Regression is a linear classifier.
     - Additionally, Naive Bayes is compared, which is a probabilistic model commonly used for text classification tasks.
- 2. **Training the models**
     - Logistic Regression and Naive Bayes models are trained using the preprocessed data that were created in the previous phase (after TF-IDF vectorization).
- 3. **Model Evaluation**
     - After training the models, their performance is evaluated on the test data. The evaluation is done using the following metrics: accuracy, precision, recall, and F1-Score. Their performance is also visualized using a confusion matrix, which shows how well the model has classified each category.
- 4. **Model Comparison**
       - By comparing the evaluation metrics of Logistic Regression and Naive Bayes, we can determine which model performs better on this classification task.
       - The confusion matrix provides further insight into the types of errors each model makes.
