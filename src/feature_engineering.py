import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# fetch data from data/processed
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# apply BoW
X_train = train_data['tweet_content'].values
y_train = train_data['sentiment'].values

X_test = test_data['tweet_content'].values
y_test = test_data['sentiment'].values

#Apply Bag of Words vectorization
vectorizer = CountVectorizer(max_features=500)
X_train_bow = vectorizer.fit_transform(X_train).toarray()
X_test_bow = vectorizer.transform(X_test).toarray()

train_df = pd.DataFrame(X_train_bow)
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow)
test_df['label'] = y_test

# store the data inside data/features
output_dir = os.path.join('data','features')
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, 'train_bow.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test_bow.csv'), index=False)