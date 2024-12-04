import pandas as pd
import re
import emoji
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch





# Load dataset
df = pd.read_csv('data/emotions.csv')

# Perform EDA
print("Missing Values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Class distribution
sns.countplot(x='label', data=df, palette="viridis")
plt.title("Class Distribution of Emotions")
plt.show()

# Remove low-percentage category (e.g., category 5 if its count is very low)
df = df[df['label'] != 5]

# Balance dataset by sampling 20k rows for each label
balanced_data = pd.DataFrame()
for label in df['label'].unique():
    sampled_data = df[df['label'] == label].sample(n=20000, random_state=42)
    balanced_data = pd.concat([balanced_data, sampled_data])

# Check new class distribution
sns.countplot(x='label', data=balanced_data, palette="viridis")
plt.title("Balanced Class Distribution")
plt.show()


# Clean text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in emoji.UNICODE_EMOJI_ENGLISH])
    text = re.sub(r'[^a-z\s]', '', text)
    return text

balanced_data['cleaned_text'] = balanced_data['text'].apply(clean_text)


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X = tfidf_vectorizer.fit_transform(balanced_data['cleaned_text'])
y = balanced_data['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
print("Naive Bayes Performance:\n", classification_report(y_test, y_pred))
