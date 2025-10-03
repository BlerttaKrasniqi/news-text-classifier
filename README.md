# Dataset: AG News Subset
- The AG News Subset dataset is used for text classification task with 4 categories of news articles: World, Business, Sports, and Sci/Tech.
- The dataset was downloaded and cached using **TensorFlow Datasets (TFDS)**. Additionally, the dataset was exported to CSV format for easier
manipulation and model training.
<img width="820" height="235" alt="image" src="https://github.com/user-attachments/assets/01a346f6-eb4e-4790-a85e-d65886ce001a" />

- Dataset link: https://www.tensorflow.org/datasets/catalog/ag_news_subset
## Data loading and Export Process
- The data is loaded into the system using **Pandas**, a python library for data manipulation.
- The training and test datasets are loaded into seperate variables train_data and test_data.
<img width="461" height="69" alt="image" src="https://github.com/user-attachments/assets/55bd3a75-28d4-419c-bec2-fc2cb9ec42f5" />

## Exporting to CSV
- The dataset is processed by concatenating the title and description of each article into a single text field.
- The processed text, along with the corresponding label, is then saved into CSV files for easy use in the future steps.
- The **write()** function is used to export the data to CSV format. It creates the CSV files and writes the data with the columns "text", and "label".
<img width="830" height="576" alt="image" src="https://github.com/user-attachments/assets/ce75e676-44b0-4872-926b-1f763d29de78" />


## Data preprocessing and cleaning
- In this phase, several data preparation and cleaning steps were performed, to ensure that the data is in a suitable format for training the machine learning model.
- Steps taken:
- 1. **Loading the Dataset**:
     - We load the training and testing datasets from CSV files using **pandas.read_csv()**.
     - The dataset consists of news articles with their corresponding labels (World, Sports, Business, Sci/Tech).
     
- 2. **Text Vectorization with TF-IDF**:
      - TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is used from **scikit-learn** to transform the text data into numerical features.
      - The **stop-words="english"** parameter removes common English stopwords (the, and, is)
      - Bigram **(ngram_range=(1,2))** is used to capture word pairs.
      - Transformation of training data and testing data using the same vectorizer.
     - Trained vectorizer is saved using **joblib** so it can be reused for prediction.
     <img width="890" height="171" alt="image" src="https://github.com/user-attachments/assets/ea8d35ae-793c-47b4-9f49-a69348d65440" />

  - 3. **Class Distribution Visualization**
       - Distribution of the news categories in the training set was visualized using a bar chart.
    <img width="842" height="161" alt="image" src="https://github.com/user-attachments/assets/84742b8f-3ed8-47d9-82b8-c17f72a633a2" />
    
       - The chart shows how many samples belong to each category (World, Sports, Business, Sci/Tech).
     <img width="796" height="575" alt="image" src="https://github.com/user-attachments/assets/4981bcb6-6763-421b-ae20-f1ea6f8bfaee" />       
  - 4. **World Cloud Visualization**
       - World Cloud is generated to visualize the most frequent words in the training set.
     <img width="820" height="105" alt="image" src="https://github.com/user-attachments/assets/89a14946-e225-448d-8237-fdf6d15c40af" />
 -  This helps understanding the key terms associated with the news articles and their categories.
       
     <img width="1078" height="598" alt="image" src="https://github.com/user-attachments/assets/8be73df9-0dba-4fac-9afd-3b6a968131a6" />
  - 5. **Top words for each category**
       - We compute the top 10 words for each news category using the TF-IDF values.
       <img width="878" height="415" alt="image" src="https://github.com/user-attachments/assets/7c504692-dcb5-4e01-a065-e450f0fd65b4" />
       - For each category (World, Sports, Business, Sci/Tech), we compute the mean TF-IDF score for each word, sort them by importance, and print the top 10 words.
       
       <img width="743" height="795" alt="image" src="https://github.com/user-attachments/assets/791c0c7e-1185-4b26-9013-319402c82362" />


  - 6. **Missing Values and Duplicates**
       - Missing values were checked both in training set and in testing set. This ensures that no data is missing in the text or label columns, which is important for accurate model training.
       <img width="707" height="158" alt="image" src="https://github.com/user-attachments/assets/fcdebfda-6af2-4e09-9d5e-7459b8e33dfc" />

       - Duplicate datas were dropped.
       <img width="739" height="79" alt="image" src="https://github.com/user-attachments/assets/7fc11626-2a63-4741-a5c8-c0779a159c6b" />

## Model training
- In this phase, the machine learning models were trained for the task of classifying news articles into four categories: World, Sports, Business, and Sci/tech. The goal of this phase is to build a classification model, evaluate its performance, and compare models to identify the best one.
- 1. **Model Selection**
     - Logistic Regression is used as the primary model for text classification. Logistic Regression is a linear classifier.
     - Additionally, Naive Bayes is compared, which is a probabilistic model commonly used for text classification tasks.
- 2. **Training the models**
     - Logistic Regression and Naive Bayes models are trained using the preprocessed data that were created in the previous phase (after TF-IDF vectorization).
     <img width="663" height="752" alt="image" src="https://github.com/user-attachments/assets/45b9c3df-b5ab-4be2-af5b-d5c37ece03a2" />

- 3. **Model Evaluation**
     - After training the models, their performance is evaluated on the test data. The evaluation is done using the following metrics: accuracy, precision, recall, and F1-Score.
     <img width="529" height="570" alt="image" src="https://github.com/user-attachments/assets/ce5b2823-d1ee-482e-88b5-f4d5ca2140e1" />
     - Their performance is also visualized using a confusion matrix, which shows how well the model has classified each category.
     <img width="1874" height="746" alt="image" src="https://github.com/user-attachments/assets/eabdc608-66d4-435e-88c8-d754b79e60c3" />
     
- 4. **Model Comparison**
       - By comparing the evaluation metrics of Logistic Regression and Naive Bayes, we can determine which model performs better on this classification task.
       - The confusion matrix provides further insight into the types of errors each model makes.
- 5. **Results**
     - Logistic Regression and Naive Bayes for classifying news articles into categories. Logistic Regression performed better overall, with higher accuracy and precision. The model has been saved for future use.
## Prediction
- A prediction script is provided that defines a function **predict_category(sentence)**. This function accepts a news article (sentence), transforms it into TF-IDF features using the loaded vectorizer, and then predicts its category using the loaded model.
- The function maps the numerical prediction to the appropriate category using category_mapping.
<img width="583" height="159" alt="image" src="https://github.com/user-attachments/assets/046433fd-c326-48f6-a645-506a1e290841" />
 - The scripts allows the user to enter a news article and get a predicted category.
<img width="628" height="262" alt="image" src="https://github.com/user-attachments/assets/c75ca092-30be-4126-bb95-31e204544ee7" />
- After executing the script, the user can write a news article and get a prediction that tells in which category the article belongs.
<img width="667" height="89" alt="image" src="https://github.com/user-attachments/assets/4dfb34e6-0160-4b58-9f3a-af69feb9fe13" />

## FastAPI: News Classification API
- This projects includes a FastAPI app that provides a real-time prediction for classifying news articles into categories like World, Sports, Business, and Sci/Tech.
- 1. **Running the API**:
     - To install the dependencies, run the following command in the terminal: **pip install fastapi uvicorn**
     - To start the API server, run the following command in the terminal: **uvicorn src.main:app-reload**
- 2. **API Endpoints**
     - FastAPI generates interactive Swagger UI for testing the API.
     - **GET /**: A simple message
     <img width="1774" height="816" alt="image" src="https://github.com/user-attachments/assets/012d5f96-dfc7-49cf-bbf9-4972a049505d" />
     - **POST /predict**: Accepts a news article text and predicts its category.
     - Request: http://127.0.0.1:8000/predict
     <img width="1816" height="802" alt="image" src="https://github.com/user-attachments/assets/0705c973-fa3e-45ce-aeda-11d9134bcab9" />
     - Results after execution:
     <img width="1773" height="753" alt="image" src="https://github.com/user-attachments/assets/a2873c55-f9d5-45cb-b06c-a11b67339d28" />


