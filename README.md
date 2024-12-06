## Hi there 👋
This project will deep dive you into the NLP and LLMs tools and techniques that are most commanly used.
## Fake News Classifier
![image](https://github.com/user-attachments/assets/29307c54-7194-483b-ad4c-5bf0915c5a76)

## Project Overview
The Fake News Classifier is a machine learning project aimed at identifying and categorizing news articles as either real or fake. By leveraging Natural Language Processing (NLP) techniques and machine learning models, the project preprocesses textual data and predicts its authenticity with high accuracy.

# 📰 Fake News Classifier  
![Python](https://img.shields.io/badge/python-3.x-blue.svg)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-green)  
![NLP](https://img.shields.io/badge/NLP-Enabled-orange)  
![Accuracy](https://img.shields.io/badge/Accuracy-90%25-brightgreen)

## 🚀 Project Overview  
Fake News Classifier is a **Machine Learning-based NLP project** designed to classify news articles as *Real* or *Fake*. By employing advanced **Natural Language Processing (NLP)** techniques and supervised learning algorithms, the model delivers an impressive **accuracy of 90%**.

### 🌟 Key Features  
- **Pre-processing:** NLTK, Regular Expressions, Stemming, Lemmatization, TF-IDF, Bag of Words (BoW), Count Vectorizer.  
- **Models Used:**  
  - Data Mining: **Porter Stemmer**  
  - Classification: **Multinomial Naïve Bayes**  
- **Dataset:** Custom or publicly available datasets with **5000 max features** for feature extraction.  
- **Performance:** Achieves **90% accuracy** on the test set.  

## 🛠️ Tools and Libraries  
- Python  
- Pandas, NumPy, Matplotlib  
- Scikit-learn  
- NLTK  

---

## 🧪 Methodology  
1. **Data Preprocessing:**  
   - Cleaned and tokenized the text using **Regular Expressions**.  
   - Removed stop words and punctuations.  
   - Applied **Stemming** and **Lemmatization** for word normalization.  
   - Extracted features using **TF-IDF**, **BoW**, and **Count Vectorizer**.  

2. **Model Training:**  
   - Utilized **Porter Stemmer** for feature mining.  
   - Applied **Multinomial Naïve Bayes** for classification.  

3. **Evaluation:**  
   - Tested on a separate dataset.  
   - Visualized results with graphs for accuracy, precision, and recall.

---

## 📊 Visualizations  
Here are some visual insights into the project's performance:
A classification matrix, also known as a confusion matrix, is a table used to evaluate the performance of a classification model. It compares the predicted labels from the model with the actual labels (true values) from the data. The confusion matrix provides a summary of prediction results and is typically used for binary and multi-class classification problems.

For binary classification, the confusion matrix looks like this:

Predicted Positive (1)	Predicted Negative (0)
Actual Positive (1)	True Positive (TP)	False Negative (FN)
Actual Negative (0)	False Positive (FP)	True Negative (TN)
Where:

True Positive (TP): The number of instances where the model correctly predicted the positive class.
False Positive (FP): The number of instances where the model incorrectly predicted the positive class (Type I error).
True Negative (TN): The number of instances where the model correctly predicted the negative class.
False Negative (FN): The number of instances where the model incorrectly predicted the negative class (Type II error).
From the confusion matrix, several important performance metrics can be derived, such as:

![image](https://github.com/user-attachments/assets/64c58cf9-52c5-4072-9c5b-901d189b1ab7)

1. *Matrix shows classification stats for PassiveModel*

![image](https://github.com/user-attachments/assets/48079c39-c1c2-4387-b286-ada85bc70664)

2. *Matrix shows classification stats for Multinomial Naive bayes Model*

![image](https://github.com/user-attachments/assets/cb50d0e9-30bd-4f16-a3c7-febd0f773fbc)

*Distribution of top features extracted from the dataset.*

---

## 📂 Project Structure  
```bash
Fake-News-Classifier/
├── data/                  # Dataset files  
├── notebooks/             # Jupyter notebooks for EDA and development  
├── src/                   # Python source files  
│   ├── preprocessing.py   # Data preprocessing code  
│   ├── train_model.py     # Model training script  
│   └── predict.py         # Prediction script  
├── requirements.txt       # Required libraries  
├── README.md              # Project documentation  
└── results/               # Evaluation results and graphs  
<!--
**narendrasingh125/narendrasingh125** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
