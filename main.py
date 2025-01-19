#%% md
# ### Emotion Analysis
#%%
# Install necessary libraries for text processing, machine learning, and deep learning
!pip install emoji  # For handling emojis in text
!pip install contractions  # For expanding contractions in text
!pip install tensorflow  # TensorFlow, includes Keras
!pip install tqdm  # For showing progress bars in loops

# Install the latest version of tf-keras (usually covered by tensorflow install)
!pip install tf-keras  # Only necessary if you want a specific version

# Optional: Specific installs for Keras if needed separately
!pip install keras  # (TensorFlow already includes Keras)

# Uncomment if you haven't downloaded necessary NLTK corpora yet
# !pip install nltk  # In case NLTK isn't installed
# import nltk
# nltk.download('stopwords')  # For stopwords
# nltk.download('punkt')  # For tokenization
#%%

#%% md
# #### Import Libraries
#%%
# Data manipulation and cleaning
import pandas as pd
import numpy as np
import re
import emoji
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from collections import Counter

# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Machine Learning models and metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc


# Deep Learning (Keras & TensorFlow)
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Word Embeddings (Gensim, FastText, Word2Vec)
import gensim
from gensim.models import Word2Vec, FastText

# Transformers
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments

# Progress bar
from tqdm import tqdm  # Progress bar for loops
#%% md
# #### Load and Explore Data
#%%
def load_data(file_path):
    """Load the dataset from a file."""
    return pd.read_csv(file_path)
#%%
def explore_data(df, text_column='text', label_column='label'):
    """Perform an extensive exploration of the dataset to check data cleanliness."""

    print("\n--- Basic Information ---")
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")

    print("\n--- Class Distribution ---")
    if label_column in df.columns:
        print(df[label_column].value_counts())

        # Plot class distribution
        plt.figure(figsize=(8, 4))
        df[label_column].value_counts().plot(kind='bar', hue='label', color='skyblue', edgecolor='black')
        plt.title("Class Distribution")
        plt.xlabel("Emotion Labels")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print(f"No '{label_column}' column found!")

    print("\n--- Text Analysis ---")
    empty_texts = df[text_column].isnull().sum() + df[text_column].str.strip().eq('').sum()
    print(f"Empty or Blank Texts: {empty_texts}")

    # Checking text length distribution
    df['text_length'] = df[text_column].apply(lambda x: len(str(x).split()))
    print(f"Average Text Length: {df['text_length'].mean():.2f} words")
    print(f"Max Text Length: {df['text_length'].max()} words")
    print(f"Min Text Length: {df['text_length'].min()} words")

    # Checking for punctuation
    punctuations = df[text_column].apply(lambda x: len(re.findall(r'[^\w\s]', str(x))))
    print(f"Average Punctuation Count per Entry: {punctuations.mean():.2f}")

    # Check for emojis
    emojis_count = df[text_column].apply(lambda x: len(emoji.emoji_list(str(x))))
    print(f"Average Emoji Count per Entry: {emojis_count.mean():.2f}")

    # Checking for stop words
    stop_words = set(stopwords.words('english'))
    stop_word_counts = df[text_column].apply(lambda x: len([word for word in str(x).split() if word.lower() in stop_words]))
    print(f"Average Stop Words per Entry: {stop_word_counts.mean():.2f}")

    # Checking for numeric-only text (Fixed issue)
    numeric_texts = df[text_column].apply(lambda x: str(x).strip().isdigit()).sum()
    print(f"Entries with Only Numbers: {numeric_texts}")

    # Checking for excessive repeated characters
    repeated_char_counts = df[text_column].apply(lambda x: len(re.findall(r'(.)\1{2,}', str(x))))
    print(f"Average Excessive Repeated Characters per Entry: {repeated_char_counts.mean():.2f}")

    print("\n--- Recommendations ---")
    recommendations = []
    if empty_texts > 0:
        recommendations.append(f"Remove or handle {empty_texts} empty or blank entries.")
    if df.duplicated().sum() > 0:
        recommendations.append("Remove duplicate rows.")
    if emojis_count.sum() > 0:
        recommendations.append(f"Consider handling {emojis_count.sum()} emojis (e.g., replace with words or remove).")
    if punctuations.mean() > 0:
        recommendations.append("Remove or handle punctuation marks appropriately.")
    if numeric_texts > 0:
        recommendations.append(f"Consider removing {numeric_texts} entries containing only numbers.")
    if repeated_char_counts.mean() > 0:
        recommendations.append("Normalize excessive repeated characters (e.g., 'looooove' → 'love').")

    if recommendations:
        print("\n".join(recommendations))
    else:
        print("The dataset appears clean!")

#%%
#Drop empty rows from our data
def drop_empty_rows(df):
    """Drop rows with empty text values."""
    return df.dropna(subset=['text']).reset_index(drop=True)
#%% md
# #### Preprocess Text
#%%
def remove_emojis(text):
    """Remove emojis from text."""
    return emoji.replace_emoji(text, replace="")

def expand_contractions(text):
    """Expand contractions like don't -> do not"""
    return contractions.fix(text)

def remove_duplicates(df, column='text'):
    """Remove duplicate rows based on a specific column."""
    before = df.shape[0]  # Get the initial number of rows
    df = df.drop_duplicates(subset=[column]).reset_index(drop=True)  # Remove duplicates
    after = df.shape[0]  # Get the number of rows after removal

    return df

def remove_punctuation_and_symbols(text):
    """Remove punctuation, special characters, and standalone numbers (not in words)."""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove standalone numbers
    return text

def normalize_repeated_characters(text):
    """Normalize words with excessive character repetition (e.g., loooove -> love)."""
    return re.sub(r'(.)\1{2,}', r'\1', text)


def preprocess_text(text):
    """Clean and preprocess text."""
    text = remove_emojis(text)
    text = expand_contractions(text)
    text = remove_punctuation_and_symbols(text)
    text = normalize_repeated_characters(text)
    tokens = word_tokenize(text.lower())  # Tokenize first, then lowercase

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_text_stemming(text):
    """Clean and preprocess text using Stemming."""
    text = remove_emojis(text)
    text = expand_contractions(text)
    text = remove_punctuation_and_symbols(text)
    text = normalize_repeated_characters(text)
    tokens = word_tokenize(text.lower())  # Tokenize first, then lowercase

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def apply_preprocessing(df):
    """Apply preprocessing to the text column."""
    df['text'] = df['text'].apply(preprocess_text)
    return df

def apply_preprocessing_stemming(df):
    """Apply stemming-based preprocessing to the text column."""
    df['text_stemmed'] = df['text'].apply(preprocess_text_stemming)
    return df
#%% md
# ### Measuring Performance and Effectiveness
#%%
# Function to measure execution time
def measure_time(func, df):
    start_time = time.time()
    df = func(df)
    end_time = time.time()
    execution_time = end_time - start_time
    return df, execution_time
#%% md
# #### Feature Engineering
#%%
def create_features_Tfidf(corpus):
    """Convert text into numerical representations."""
    vectorizer = TfidfVectorizer(max_features=5000)
    features = vectorizer.fit_transform(corpus)
    return features, vectorizer

def create_features_CountVectorizer(corpus):
    """Convert text into numerical representations."""
    vectorizer = CountVectorizer(max_features=5000)
    features = vectorizer.fit_transform(corpus)
    return features, vectorizer

# Create features using Word2Vec
def create_features_Word2Vec(corpus):
    """Convert text into numerical representations using Word2Vec."""
    tokenized_corpus = [doc.split() for doc in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    # Calculate feature vectors as the mean of word vectors for each document
    features = np.array([
        np.mean([
            model.wv[word] for word in doc if word in model.wv
        ] or [np.zeros(100)], axis=0) for doc in tokenized_corpus
    ])
    return features, model

def batch_bert_embeddings(corpus, batch_size=256):
    """Efficiently process BERT embeddings in batches with progress tracking."""
    features = []
    num_batches = len(corpus) // batch_size + 1  # Calculate total batches

    print(f"Processing {len(corpus)} texts in {num_batches} batches...")

    for i in tqdm(range(0, len(corpus), batch_size), desc="Extracting BERT Features"):
        batch = corpus[i:i + batch_size].tolist()  # Convert batch to list
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)  # Move to GPU

        with torch.no_grad():
            outputs = model(**inputs)

        batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token representation
        features.append(batch_features)

    print("BERT Feature Extraction Completed!")
    return np.vstack(features)  # Combine all batches

#%% md
# ### Exploratory data analysis(EDA)
#%% md
# #### Model Training
#%% md
# ##### Logistic Regression
#%%
def train_logistic_regression(X, y):
    """Train and evaluate a Logistic Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("\n--- Logistic Regression Evaluation ---")
    evaluate_model(model, X_test, y_test)
    return model
#%% md
# ##### Random Forest
#%%
def train_random_forest(X, y):
    """Train and evaluate a Random Forest Classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("\n--- Random Forest Evaluation ---")
    evaluate_model(model, X_test, y_test)
    return model
#%% md
# ##### Support Vector Machine
#%%
def train_svm(X, y):
    """Train and evaluate a Support Vector Machine (SVM)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model = SVC(probability=False)

    """Train and evaluate a fast Support Vector Machine (SVM) using LinearSVC."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearSVC(dual=False)  # 🚀 Much faster than SVC(kernel='linear')
    model.fit(X_train, y_train)

    # model.fit(X_train, y_train)
    print("\n--- SVM Evaluation ---")
    evaluate_model(model, X_test, y_test)
    return model

#%% md
# #### Evaluate Model
#%%
#Evaluation Function

def evaluate_model(model, X_test, y_test):
    try:
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC (for binary classification)
except Exception as e:
    print(f"Error in prediction: {e}")


    if y_pred is None or y_prob is None:
        print("Model did not produce valid predictions.")

    """Evaluate the performance of a trained model."""
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # Calculate Precision, Recall, F1-Score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print Precision, Recall, and F1-score
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Visualize Confusion Matrix using a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

        # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
#%% md
# #### Predict
#%%
def predict_emotion(model, text, vectorizer):
    """Predict the emotion of a single input text."""
    processed_text = preprocess_text(text)
    features = vectorizer.transform([processed_text])
    return model.predict(features)

#%%
emotions_df = load_data('data/emotions.csv')
explore_data(emotions_df, text_column='text')

#%% md
# #### Class Distribution (Imbalance Visualization)
#%%
# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=emotions_df, hue='label', dodge=False, palette='viridis', legend=False)
plt.title("Class Distribution of Labels", fontsize=14)
plt.xlabel("Labels (Emotions)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.show()
#%% md
# #### Remove duplicated data
#%%
emotions_df = remove_duplicates(emotions_df, column='text')  # Remove duplicates first
emotions_df = emotions_df[emotions_df['text'].str.strip() != ""].reset_index(drop=True)

#%% md
# ### Undersampling data
#%%
# Display the first few rows
emotions_df.info()
#%%
# Undersampling 14k rows from each category 0, 1, 2, 3, 4, 5
# Find the least frequent class count
min_count = emotions_df['label'].value_counts().min()  # Get smallest class size
print(f"Undersampling to {min_count} rows per class")


# Apply undersampling to match the smallest class
undersample_dataset = pd.concat([
    emotions_df[emotions_df['label'] == label].sample(n=min_count, random_state=42)
    for label in emotions_df['label'].unique()
])


# Display new class distribution
print("\nNew Class Distribution After Undersampling:")
print(undersample_dataset['label'].value_counts())

#%%
explore_data(undersample_dataset, text_column='text')  # Adjust 'text' if your column name differs
#%% md
# ### Measuring Lemmatization/Stemming Processing Time
#%%
# Measure Lemmatization Time
df, lemmatization_time = measure_time(apply_preprocessing, undersample_dataset)
print(f"Lemmatization Time: {lemmatization_time:.4f} seconds")

# Measure Stemming Time
df, stemming_time = measure_time(apply_preprocessing_stemming, undersample_dataset)
print(f"Stemming Time: {stemming_time:.4f} seconds")
#%% md
# ## Exploratory Data Analysis (EDA)
#%% md
# #### Class Distribution of Emotion Labels After Undersampling
#%%
# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=undersample_dataset, hue='label', dodge=False, palette='viridis', legend=False)
plt.title("Class Distribution of Labels After Undersampling", fontsize=14)
plt.xlabel("Labels (Emotions)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.show()
#%% md
# #### Most Frequent Words in Text Data (Excluding Stopwords)
#%%

stop_words = set(stopwords.words('english'))

# Count word occurrences
word_counts = Counter(" ".join(undersample_dataset['text']).split())
filtered_words = {word: count for word, count in word_counts.items() if word.lower() not in stop_words}

# Plot top 20 words
plt.figure(figsize=(10,5))
plt.barh(list(filtered_words.keys())[:20], list(filtered_words.values())[:20], color='skyblue')
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Words", fontsize=12)
plt.title("Most Frequent Words in Text Data", fontsize=14)
plt.gca().invert_yaxis()
plt.show()

#%% md
# #### Overall Text Length Distribution
#%%
undersample_dataset['text_length'] = undersample_dataset['text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10,5))
plt.hist(undersample_dataset['text_length'], bins=30, color='skyblue', edgecolor='black')
plt.title("Text Length Distribution", fontsize=14)
plt.xlabel("Number of Words", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()
#%% md
# #### Word Count Distribution by Emotion
#%%
# !pip install plotly
import plotly.express as px

# Create a column for text length (word count)
undersample_dataset['word_count'] = undersample_dataset['text'].apply(lambda x: len(str(x).split()))

# Box plot for word count distribution across emotion labels
fig = px.box(undersample_dataset, y="word_count", color="label", template="plotly_white")

# Show the plot
fig.show()

#%% md
# #### Class Distribution of Emotion Labels
#%%
# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=undersample_dataset, hue='label', dodge=False, palette='viridis', legend=False)
plt.title("Class Distribution of Labels", fontsize=14)
plt.xlabel("Labels (Emotions)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.show()
#%% md
# #### Emotion-Specific Keywords
#%%
# TF-IDF analysis
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(undersample_dataset['text'])

# Convert to DataFrame
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df['emotion'] = undersample_dataset['label']

# Find top keywords for each emotion
top_keywords = {}
for emotion in tfidf_df['emotion'].unique():
    emotion_data = tfidf_df[tfidf_df['emotion'] == emotion]
    top_keywords[emotion] = emotion_data.mean(axis=0).sort_values(ascending=False).head(10)

print(top_keywords)

#%% md
# #### Heatmap for TF-IDF Top Keywords
#%%
# Heatmap for TF-IDF scores
tfidf_top_words = tfidf_df.drop('emotion', axis=1).mean(axis=0).sort_values(ascending=False).head(30)
plt.figure(figsize=(12, 8))
sns.heatmap(pd.DataFrame(tfidf_top_words).T, annot=True, cmap='viridis')
plt.title("TF-IDF Top Keywords Heatmap")
plt.show()

#%%
emotions_df = apply_preprocessing(undersample_dataset)
#%%
emotions_string = ' '.join(emotions_df['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(emotions_string)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud: Visualizing the most common words")

plt.show()
#%% md
# ---
#%% md
# ### Feature Extraction using Text Vectorization Techniques
#%%
#create_features_Tfidf
XTfidf, vectorizer = create_features_Tfidf(emotions_df['text'])
XWord2Vec, vectorizer = create_features_Word2Vec(emotions_df['text'])
XCountVectorizer, vectorizer = create_features_CountVectorizer(emotions_df['text'])


y = emotions_df['label']  # Assuming the label column is named 'label'

#%%
# XBERT, vectorizer = create_features_BERT(emotions_df['text'])  # 'text' is the name of your text column
# Run BERT embedding extraction

XBERT = batch_bert_embeddings(emotions_df['text'], batch_size=256)  # Lower batch_size if memory issues occur



#%%
# Extract and display the shape of each feature representation
print(f"TF-IDF Shape: {XTfidf.shape}")
print(f"Word2Vec Shape: {XWord2Vec.shape}")
print(f"Word2Vec Shape: {XCountVectorizer.shape}")
# print(f"BERT Embeddings Shape: {XBERT.shape}")

#%%
logistic_model = train_logistic_regression(XTfidf, y)
#%%

#%%
logistic_model = train_logistic_regression(XWord2Vec, y)
#%%
logistic_model = train_logistic_regression(XCountVectorizer, y)
#%%
random_forest_model = train_random_forest(XTfidf, y) #XTfidf XWord2Vec XCountVectorizer
#%%
svm_model = train_svm(XTfidf, y) #XWord2Vec XCountVectorizer
#%%
svm_model = train_svm(XCountVectorizer, y)
#%%
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(XWord2Vec, y, test_size=0.2, random_state=42)

#%%
# Reshape the data for LSTM: (samples, timesteps, features)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#%% md
# ### SimpleRNN model
#%%
# Define the SimpleRNN model
n_classes = len(np.unique(y))

rnn_model = Sequential()
rnn_model.add(Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))  # Correct input shape

rnn_model.add(SimpleRNN(256, activation='tanh', return_sequences=True))  # First RNN layer
rnn_model.add(Dropout(0.2))  # Reduce overfitting

rnn_model.add(SimpleRNN(128, activation='tanh', return_sequences=True))  # Second RNN layer
rnn_model.add(Dropout(0.2))

rnn_model.add(SimpleRNN(64, activation='tanh'))  # Final RNN layer
rnn_model.add(Dropout(0.2))

rnn_model.add(Dense(50, activation='relu'))  # Dense hidden layer
rnn_model.add(Dense(n_classes, activation='softmax'))  # Output layer for multi-class classification

# Compile the model
# rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Compile the model with Adam optimizer and a learning rate
rnn_model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

rnn_model.summary()
#%%
# Train the model
rnn_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=1)

#%%
# Evaluate the model
loss, accuracy = rnn_model.evaluate(X_test_reshaped, y_test, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
#%%

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = rnn_model.fit(X_train_reshaped, y_train,
                         epochs=10, batch_size=32,
                         validation_data=(X_test_reshaped, y_test),
                         callbacks=[early_stopping], verbose=1)




# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 6))


# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
#%%
print(np.mean(X_word2vec, axis=0))
print("---------------------------- ----------------------------")
print(np.std(X_word2vec, axis=0))
#%% md
# ### LSTM
#%%
# Define the LSTM model
n_classes = len(np.unique(y))

lstm_model = Sequential()
# lstm_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Correct input shape
lstm_model.add(Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))  # Correct input shape

lstm_model.add(LSTM(256, activation='tanh', return_sequences=True))

lstm_model.add(LSTM(128, activation='tanh', return_sequences=True))  # First LSTM
lstm_model.add(Dropout(0.2))  # Reduced dropout for better learning

lstm_model.add(LSTM(64, activation='tanh'))  # Second LSTM
lstm_model.add(Dropout(0.2))  # Reduced dropout


lstm_model.add(Dense(50, activation='relu'))
lstm_model.add(Dense(n_classes, activation='softmax')) # Output layer for multi-class classification


lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.summary()

#%%
# Train the model
lstm_model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, verbose=1)

#%%
# Evaluate the model
loss, accuracy = lstm_model.evaluate(X_test_reshaped, y_test, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
#%%
# Define model names and corresponding performance metrics
models = ["Logistic Regression", "Random Forest", "SVM"]
accuracy = [0.93, 0.93, 0.94]
f1_scores = [0.93, 0.93, 0.94]
#%%
# Plot Model Accuracy Comparison
plt.figure(figsize=(10, 6))
plt.bar(models, accuracy, color=['blue', 'green', 'red'], alpha=0.7)
plt.ylim(0.9, 1.0)  # Set y-axis limit for better visualization
plt.title("Model Accuracy Comparison", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
#%%
# Plot Model Macro F1-score Comparison
plt.figure(figsize=(10, 6))
plt.bar(models, f1_scores, color=['purple', 'orange', 'cyan'], alpha=0.7)
plt.ylim(0.9, 1.0)  # Set y-axis limit
plt.title("Model Macro F1-score Comparison", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("F1-score", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
#%%
# Early stopping callback: Stop training if the validation loss doesn't improve for 5 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Train the model with validation data and early stopping
history = lstm_model.fit(X_train_reshaped, y_train,
                         epochs=10, batch_size=32,
                         validation_data=(X_test_reshaped, y_test),
                         callbacks=[early_stopping], verbose=1)

# Plot training and validation loss/accuracy
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
#%% md
# ### BiDirectional LSTM
#%%

#%% md
# ## Trying BERT
#%%
XBERT, vectorizer = create_features_BERT(emotions_df['text'])  # 'text' is the name of your text column
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
