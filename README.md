## Hi there 👋
This project will deep dive you into the NLP and LLMs tools and techniques that are most commanly used.
## Fake News Classifier
![image](https://github.com/user-attachments/assets/29307c54-7194-483b-ad4c-5bf0915c5a76)

## Project Overview
The Fake News Classifier is a machine learning project aimed at identifying and categorizing news articles as either real or fake. By leveraging Natural Language Processing (NLP) techniques and machine learning models, the project preprocesses textual data and predicts its authenticity with high accuracy.

## Project Objectives
To preprocess news articles efficiently using NLP techniques.
To classify news articles using a machine learning algorithm.
To achieve high accuracy and provide insights through graphical analysis of the results.  
Tech Stack
Programming Language: Python
Libraries/Frameworks:
NLP Techniques: NLTK, Regular Expressions
Data Preprocessing: TF-IDF, Bag of Words (BoW), Count Vectorizer
Models: Porter Stemmer, Multinomial Naïve Bayes
Data Preprocessing Steps
Text Cleaning: Removed punctuations, special characters, and irrelevant data using regular expressions.
Tokenization: Split text into individual words.
Stemming: Reduced words to their root form using the Porter Stemmer.
Lemmatization: Standardized words to their base dictionary forms.
Feature Extraction:
Implemented TF-IDF (Term Frequency-Inverse Document Frequency) for feature representation.
Used Bag of Words (BoW) and Count Vectorizer with a maximum of 5000 features for text vectorization.
Model Implementation
Algorithm: Multinomial Naïve Bayes
Dataset: Preprocessed news dataset with labeled data (real or fake).
Performance: Achieved an accuracy of approximately 90% on the test data.
Results and Visualization
Graphs Included:
Confusion Matrix: Showcases true positives, true negatives, false positives, and false negatives.
Accuracy and Loss Curve: Tracks model performance during training.
Feature Importance: Highlights the top keywords contributing to classification decisions.
Example Graphs:
Accuracy Curve: Displays the trend of training vs. validation accuracy over epochs.
Word Frequency Bar Chart: Highlights the most frequent words in real and fake news categories.
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

![Accuracy Graph](https://via.placeholder.com/800x400?text=Accuracy+Graph)  
*Graph showcasing model accuracy over iterations.*

![Feature Distribution](https://via.placeholder.com/800x400?text=Feature+Distribution)  
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
