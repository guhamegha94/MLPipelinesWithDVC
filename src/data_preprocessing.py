import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# transform data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab') # Download punkt for tokenization

# fetch data from data/raw
train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

def lemmatization(text):
    """Lemmatize the input text."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def remove_stopwords(text):
    """Remove stopwords from the input text."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)                     # mentions
    text = re.sub(r"#", "", text)                        # hashtag symbol
    text = re.sub(r"\d+", "", text)                      # numbers
    text = re.sub(r"[^\w\s]", "", text)                  # punctuation
    text = re.sub(r"\s+", " ", text).strip()             # extra spaces
    return text

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    # First, ensure tweet_content is string type and handle NaN
    df['tweet_content'] = df['tweet_content'].fillna('').astype(str)
    df.tweet_content = df.tweet_content.apply(lambda content: clean_tweet(content))
    df.tweet_content = df.tweet_content.apply(lambda content: remove_stopwords(content))
    df.tweet_content = df.tweet_content.apply(lambda content: lemmatization(content))
    return df

train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)

# store the data inside data/processed
data_path = os.path.join('data','processed')

os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'), index=False)
test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'), index=False)

