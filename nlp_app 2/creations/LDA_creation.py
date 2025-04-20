import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from gensim import models, corpora 
from itertools import combinations
import pyLDAvis
import pyLDAvis.gensim_models 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import plotly.express as px 
import streamlit as st
import spacy
import string


nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
stop_words= list(stop_words) + ['card','I']
def remove_stopwords(text):
    
    # Convert to lowercase converts to tokens so easy to get lemmatize 
    doc = nlp(text.lower())
    cleaned_words = [
    token.lemma_  # Lemmatization
    for token in doc
    if token.lemma_ not in string.punctuation and token.lemma_ not in stop_words
]

    return " ".join(cleaned_words)


def remove_stopwords_woLemma(text):# same but not lemmatilization
    
    # Convert to lowercase converts to tokens so easy to get lemmatize 
    doc = nlp(text.lower())
    cleaned_words = [
    token.text  # Lemmatization
    for token in doc
    if token.text not in string.punctuation and token.text not in stop_words
    
]

    return " ".join(cleaned_words)


# Hyperparameter Tuning for Optimal Topics
def find_optimal_topics(corpus,dictionary,texts,min_topics=5, max_topics=25, step=5):
    coherence_values = []
    all_models = []
    
    for num_topics in range(min_topics, max_topics+1, step):
        model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            alpha='auto',
            eta='auto',
            passes=15
        )
        
        coherence = models.CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        ).get_coherence()
        
        coherence_values.append(coherence)
        all_models.append(model)
        print(f"Topics: {num_topics} | Coherence: {coherence:.3f}")
    
    optimal_index = np.argmax(coherence_values)
    return all_models[optimal_index], coherence_values[optimal_index]

# Stability Analysis with Jaccard Similarity
def jaccard_similarity(topic1, topic2):
    """Calculate Jaccard similarity between two topics"""
    set1 = set([word for word, _ in topic1])
    set2 = set([word for word, _ in topic2])
    return len(set1 & set2) / len(set1 | set2)

def stability_analysis(lda_model,corpus,dictionary,num_runs=5):
    all_topics = []
    for _ in range(num_runs):
        model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=lda_model.num_topics,
            alpha='auto',  # Automatic asymmetric alpha
            eta='auto',    # Automatic asymmetric eta
            passes=15,
        )
        
        all_topics.append([model.show_topic(tid, topn=10) for tid in range(model.num_topics)]) #getting words and weights that contribute to topic
    
    # Compare topic words across runs
    similarities = []
    for run1, run2 in combinations(all_topics, 2):
        run_sim = [max(jaccard_similarity(t1, t2) for t2 in run2) for t1 in run1]
        similarities.append(np.mean(run_sim))
    
    return np.mean(similarities) # getting average jaccard similarity per run





train=pd.read_csv('trainSA.csv')
test=pd.read_csv('testSA.csv')
data=pd.concat([train,test]).reset_index()
data=data.drop(columns=['index'])


texts = [document.lower().split() for document in data['text'].apply(remove_stopwords_woLemma)]
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below = 4, no_above= .9) # must appear 4 times in text and not be in 90% of the comments 
corpus = [dictionary.doc2bow(text) for text in texts]

print('running model')
model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=45,
        alpha='auto',
        eta='auto',
        passes=15
    )
print('saved model')
model.save('.nlp_app/tabs/best_lda_model')