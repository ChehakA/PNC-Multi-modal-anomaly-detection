import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from textstat import flesch_reading_ease, gunning_fog
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from wordcloud import WordCloud

import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
        return ' '.join(tokens)
    return "" 
def load_data():
    data_train = pd.read_csv('../tabs/trainSA.csv')
    data_test = pd.read_csv('../tabs/testSA.csv')
    return data_train, data_test
data_train, data_test = load_data()
data_train['clean_text'] = data_train['text'].apply(preprocess_text)





# making reduced data 
print('training starting')
vectorizer = TfidfVectorizer(max_features=100)
train_features = vectorizer.fit_transform(data_train['clean_text'])
train_features_scaled = train_features.toarray()

reducer = UMAP(random_state=42)
train_umap = reducer.fit_transform(train_features_scaled)
np.save('../tabs/train_umap.npy', train_umap)
print('saved train_umap')

print('training tsne')
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
train_tsne = tsne.fit_transform(train_umap)
np.save('../tabs/train_tsne.npy', train_tsne)
print('saved train_tsne')

print('tsne_results')
tsne_results = TSNE(n_components=2, random_state=42).fit_transform(train_features_scaled)
np.save('../tabs/tsne_results.npy', tsne_results)
print('saved tsne_results')



