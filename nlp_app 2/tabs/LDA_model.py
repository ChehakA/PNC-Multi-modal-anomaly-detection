import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pyLDAvis
import pyLDAvis.gensim_models 
import plotly.express as px 
import streamlit as st
import streamlit.components.v1 as components
import spacy 
import string
from gensim import models, corpora 
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def render():

    nlp = spacy.load("en_core_web_sm")
    stop_words = nlp.Defaults.stop_words
    stop_words= list(stop_words) + ['card','I']
  
    def remove_stopwords_woLemma(text):# same but not lemmatilization
        
        # Convert to lowercase converts to tokens so easy to get lemmatize 
        doc = nlp(text.lower())
        cleaned_words = [
        token.text  # Lemmatization
        for token in doc
        if token.text not in string.punctuation and token.text not in stop_words
        
    ]

        return " ".join(cleaned_words)

    @st.cache_data
    def load_data():
        train=pd.read_csv('tabs/trainSA.csv')
        test=pd.read_csv('tabs/testSA.csv')
        data=pd.concat([train,test]).reset_index()
        data=data.drop(columns=['index'])
        return data 
    
    @st.cache_data
    def load_corpus(data):
        texts = [document.lower().split() for document in data['text'].apply(remove_stopwords_woLemma)]
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below = 4, no_above= .9) # must appear 4 times in text and not be in 90% of the comments 
        corpus = [dictionary.doc2bow(text) for text in texts]
        return texts,dictionary,corpus

   

    
    data = load_data()

    with st.expander(" data distribution", expanded=True):
        fig, ax = plt.subplots()
        ax.bar(data['category'].value_counts().index ,data['category'].value_counts());
        ax.set_xticks(data['category'].value_counts().index[::5],);
        ax.set_xticklabels(
        data['category'].value_counts().index[::5],
        rotation=45,
        ha='right'
    )
        # ax.set_xticks(rotation=45, ha='right');
        ax.set_xlabel('Text category');
        ax.set_ylabel('Data amount');
        ax.set_title('Amount of data per category');
        st.pyplot(fig)
        



    data['char_count'] = data['text'].str.len()
    data['word_count'] = data['text'].str.split().str.len()
    with st.expander(" Char frequency ", expanded=True):
        fig,ax = plt.subplots()
        ax.hist(data['char_count']);
        ax.set_xlabel('character count');
        ax.set_ylabel('frequency');
        ax.set_title('char count')
        st.pyplot(fig)


   

    with st.expander(" Word Count ", expanded=True):
        fig,ax = plt.subplots()
        ax.hist(data['word_count']);
        ax.set_xlabel('word count');
        ax.set_ylabel('frequency');
        ax.set_title('word count');
    

    texts,dictionary,corpus=load_corpus(data)
  

    with st.expander("LDA Module", expanded=True):
        # run this model then save it and import it 
        st.header('Notes')
        st.subheader('Coherence')

        st.markdown("""
    I'm using the **Coherence Score** to find the optimal number of topics based on word similarity per topic.

    - A **lower** coherence score means worse topic/word relationships.
    - A **higher** coherence score means the words relate better semantically, making the topics more human-interpretable.

    📖 [Good breakdown on StackOverflow](https://stackoverflow.com/questions/54762690/evaluation-of-topic-modeling-how-to-understand-a-coherence-value-c-v-of-0-4)

    Below is the cohernce score based on the number of topics selected for our data.
    Typically the highest cohernce was between 45 and 55 clusters and
    """)

        st.image("tabs/topics-score.png")
        st.subheader('Perplexity')
        st.markdown("""
    Measures the models ability to predict unseen documents to a topic 
    - The **lower** the perplexity score the better
    - The **higher** the perplexity score the worse the model is
        """)
        
        st.subheader(' Model params and make up ')
        
        st.markdown("""
    - Removed stop words without lemmatizing data 
    - Dictionary only contains text that appeared 4 times  and not in 90% of the comments
    - Used 45 topics based on cohernce chart 
        """)

        st.subheader('Chart analysis explanaition')

        st.markdown("""
    Right Side 
    
    - Red Bars: Show term frequency within the selected topic
    
    - Blue Bars: Show overall term frequency across all documents
    
    - High λ: Shows common terms that strongly represent the topic
    
    - Low λ: Reveals distinctive terms that uniquely identify the topic
    
    Comparing bars
    
    - Large gap between red and blue: Term is distinctive to this topic
    
    - Similar heights: Term appears similarly across multiple topics
    
    Left Side 
    
    - smaller circles means less data assigned to topic
    - bigger circles means more data affiliated with topic 
    - overlapping or close means similar in topic 

    [A good demo that explained how to interpret the chart](https://developer.ibm.com/tutorials/awb-lda-topic-modeling-text-analysis-python/#step-9-text-classification11)
        """)

        with st.spinner("Training LDA Model..."):
            # test to see if reading in the wirgth thing and if it's fast
            model = models.LdaModel.load("tabs/best_lda model")
            # pyLDAvis.display.DEFAULT_DIV_STYLE = "background-color:black; width:100%; height:100%;"
            with open('tabs/lda_visualization.html', 'r', encoding='utf-8') as f:
                html_string=f.read()

            # html_string = pyLDAvis.prepared_data_to_html(visual)
            # visual = pyLDAvis.gensim_models.prepare(model, corpus, dictionary, mds='mmds')
            # html_string = pyLDAvis.prepared_data_to_html(visual)
            components.html(html_string, width=10000, height=900, scrolling=True)

            st.write(f"Model perplexity (with lemmea): {model.log_perplexity(corpus):.2f}" )