import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gensim.models import Word2Vec
import numpy as np

# 1. Load data
df = pd.read_csv('data/clean/transactions.csv')
print(df.shape)

# 2. Vectorizing text data
# Bag-of-Words Representation
def bow_representation(data):
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    X_bow = vectorizer.fit_transform(data)
    X_bow = X_bow.toarray()
    return X_bow

X_bow = bow_representation(df['Description'])
print('Descriptions: Bag of Words', X_bow.shape)

# TF-IDF
def tfidf_representation(data):
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    X_bow = vectorizer.fit_transform(data)
    transformer = TfidfTransformer(smooth_idf=False)
    X_tfidf = transformer.fit_transform(X_bow)
    X_tfidf = X_tfidf.toarray()
    return X_tfidf
X_tfidf = tfidf_representation(df['Description'])
print('Descriptions: TF-IDF', X_tfidf.shape)

# Word Embeddings
def word_embeddings(data):
    sentences = [desc.split() for desc in data]
    model = Word2Vec(sentences, min_count=1)
    X_w2v = np.array([model.wv[desc] for desc in sentences], dtype=object)
    return X_w2v
X_w2v = word_embeddings(df['Description'])
print('Descriptions: Word Embeddings', X_w2v.shape)

# Date Features
df['Date'] = pd.to_datetime(df['Date'])
df['Day of Week'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month

# Encode day of week and month as numerical values
le = LabelEncoder()
df['Day of Week'] = le.fit_transform(df['Day of Week'])
df['Month'] = le.fit_transform(df['Month'])
print('Date Features:', df[['Day of Week', 'Month']].shape)

# Amount Features
scaler = StandardScaler()
df['Normalized Amount'] = scaler.fit_transform(df[['Amount']])
print('Amount Features:', df[['Normalized Amount']].shape)

# Combined Feature Vector
# Here we're using the TF-IDF features as an example, but you could replace this with BoW or Word2Vec features
# Convert X_tfidf to a DataFrame
df_tfidf = pd.DataFrame(list(X_tfidf))

# Create a new DataFrame with the other features
df_other_features = df[['Day of Week', 'Month', 'Normalized Amount']]

# Reset the index of both DataFrames to ensure they align correctly
df_tfidf.reset_index(drop=True, inplace=True)
df_other_features.reset_index(drop=True, inplace=True)

# Concatenate the DataFrames along the columns axis
features = pd.concat([df_tfidf, df_other_features], axis=1)
print('Combined Feature Vector:', features.shape)

# Save the features to a CSV file
features.to_csv('data/processed/features.csv', index=False)
