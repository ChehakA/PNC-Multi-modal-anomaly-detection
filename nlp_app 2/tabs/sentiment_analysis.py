import streamlit as st
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
from sklearn.ensemble import IsolationForest
from wordcloud import WordCloud
from hdbscan import prediction
import hdbscan
from umap import UMAP
from sentence_transformers import SentenceTransformer

@st.cache_data
def load_data():
    data_train = pd.read_csv('tabs/trainSA.csv')
    data_test = pd.read_csv('tabs/testSA.csv')
    return data_train, data_test

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
        return ' '.join(tokens)
    return ""

def contains_high_charge(text):
    amounts = re.findall(r"\$?\b\d{2,}\b", text)
    return any(int(amount.replace('$', '')) > 50 for amount in amounts)

def contains_suspicious_keywords(text):
    text = text.lower()
    keywords = [
        "unauthorized", "without purchase", "extra charge", "didn’t buy", "did not buy", "charged more",
        "unexpected charge", "lost", "never arrived", "locked out", "access denied", "can't log in",
        "didn't authorize", "fraudulent", "card blocked", "help", "scammed", "cancel subscription",
        "why was i charged", "my card was charged", "chargeback", "hacked", "pending charge"
    ]
    word_combos = [("charged", "extra"), ("fee", "applied"), ("billed", "wrong"), ("locked", "account")]

    if any(phrase in text for phrase in keywords):
        return True
    for word1, word2 in word_combos:
        if word1 in text and word2 in text:
            return True
    return False

@st.cache_data
def preprocess_dataframe(df):
    df = df.copy()
    df['clean_text'] = df['text'].apply(preprocess_text)
    return df

def create_word_cloud(text_corpus):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

def create_sentiment_chart(sentiment_df):
    sentiment_melt = sentiment_df[['neg', 'neu', 'pos']].melt(var_name='Sentiment', value_name='Score')
    chart = alt.Chart(sentiment_melt).mark_bar().encode(
        x='Sentiment:N',
        y='mean(Score):Q',
        color='Sentiment:N',
        tooltip=['Sentiment', 'mean(Score)']
    ).properties(title='Sentiment Distribution', width=600, height=400)
    return chart

def create_clustering_chart(df, x, y, color, title):
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=f'{x}:Q',
        y=f'{y}:Q',
        color=alt.Color(f'{color}:N', scale=alt.Scale(domain=['Anomaly', 'Normal'], range=['red', 'blue'])),
        tooltip=['text']
    ).properties(title=title, width=700, height=500).interactive()
    return chart

@st.cache_data
def get_cluster_models(clean_texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_features = model.encode(clean_texts.tolist())
    reducer = UMAP(random_state=42)
    train_umap = reducer.fit_transform(train_features)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, cluster_selection_method='eom', prediction_data=True)
    hdbscan_labels = clusterer.fit_predict(train_umap)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(train_features)
    distances = kmeans.transform(train_features)
    iso_forest = IsolationForest(random_state=42, contamination=0.05)
    iso_forest.fit(train_features)
    iso_labels = iso_forest.predict(train_features)
    return model, reducer, clusterer, kmeans, distances, train_features, train_umap, hdbscan_labels, kmeans_labels, iso_forest, iso_labels

def main():
    st.title("📊 Advanced Sentiment Analysis Dashboard")

    nltk.download(['punkt', 'wordnet', 'vader_lexicon', 'stopwords'], quiet=True)
    sia = SentimentIntensityAnalyzer()
    data_train, data_test = load_data()
    data_train = preprocess_dataframe(data_train)
    model, reducer, clusterer, kmeans, distances, train_features, train_umap, hdbscan_labels, kmeans_labels, iso_forest, iso_labels = get_cluster_models(data_train['clean_text'])

    # 🧪 Demo Input Box FIRST
    st.subheader("🔎 Try Your Own Text for Anomaly Detection")
    user_input = st.text_area("Enter a sentence or paragraph:", height=150)

    # Sidebar Weights
    hdbscan_weight = st.sidebar.slider("HDBSCAN Weight", 0.0, 1.0, 0.5, 0.05)
    kmeans_weight = st.sidebar.slider("KMeans Weight", 0.0, 1.0, 0.25, 0.05)
    iso_weight = st.sidebar.slider("Isolation Forest Weight", 0.0, 1.0, 0.25, 0.05)
    threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.3, 0.05)

    if st.button("Detect Anomaly"):
        if user_input.strip():
            clean_input = preprocess_text(user_input)
            input_vector = model.encode([clean_input])
            input_umap = reducer.transform(input_vector)

            # 🔹 Predictions
            hdbscan_pred, _ = prediction.approximate_predict(clusterer, input_umap)
            hdbscan_result = 'Anomaly' if hdbscan_pred[0] == -1 else 'Normal'
            kmeans_distance = kmeans.transform(input_vector).min()
            kmeans_threshold = np.percentile(distances.min(axis=1), 95)
            kmeans_result = 'Anomaly' if kmeans_distance > kmeans_threshold else 'Normal'
            iso_pred = iso_forest.predict(input_vector)[0]
            iso_score_val = iso_forest.decision_function(input_vector)[0]
            iso_result = 'Anomaly' if iso_pred == -1 else 'Normal'

            # 🔸 Scoring
            hdbscan_score = 1 if hdbscan_result == 'Anomaly' else 0
            kmeans_score = 1 if kmeans_result == 'Anomaly' else 0
            iso_score = 1 if iso_result == 'Anomaly' else 0

            weighted_score = (
                hdbscan_score * hdbscan_weight +
                kmeans_score * kmeans_weight +
                iso_score * iso_weight
            )

            rule_flagged = contains_high_charge(user_input) or contains_suspicious_keywords(user_input)

            if rule_flagged and weighted_score > 0:
                final_result = 'Anomaly'
                verdict_reason = "Rule-based flag + model support"
            elif weighted_score >= threshold or rule_flagged:
                final_result = 'Anomaly'
                verdict_reason = "Weighted score or rule-based flag"
            else:
                final_result = 'Normal'
                verdict_reason = "Below threshold and no rule trigger"

            # 🧠 Model Outputs
            st.subheader("🧠 Model Predictions:")
            col1, col2, col3 = st.columns(3)
            col1.metric("HDBSCAN", hdbscan_result)
            col2.metric("KMeans", kmeans_result)
            col3.metric("Isolation Forest", iso_result)

            st.markdown(f"""
            **Model Scores**  
            - HDBSCAN Score: `{hdbscan_score}`  
            - KMeans Score: `{kmeans_score}`  
            - Isolation Forest Score: `{iso_score}`  
            - Isolation Forest Raw Score: `{iso_score_val:.4f}`

            **Heuristic Rule Triggered:** `{rule_flagged}`  
            **Weighted Score:** `{weighted_score:.2f}` (Threshold: `{threshold}`)

            ### 🏁 **Final Verdict:** `{final_result}`  
            _Reason: {verdict_reason}_
            """)

            sentiment = sia.polarity_scores(user_input)
            compound = sentiment['compound']
            if compound >= 0.05:
                sentiment_label = "Positive 😊"
            elif compound <= -0.05:
                sentiment_label = "Negative 😠"
            else:
                sentiment_label = "Neutral 😐"

            st.subheader("💬 Sentiment Behind Your Input")
            st.markdown(f"**Sentiment:** `{sentiment_label}`")
            st.markdown(f"**Scores:**")
            st.json(sentiment)

            # UMAP Visualization
            st.markdown("### 📍 Visualizing Your Input in Context")

            plot_df = pd.DataFrame(train_umap, columns=["x", "y"])
            plot_df["label"] = np.where(np.array(iso_labels) == -1, "Anomaly", "Normal")

            user_point = pd.DataFrame({
                "x": [input_umap[0][0]],
                "y": [input_umap[0][1]],
                "label": ["Your Input"]
            })

            plot_df = pd.concat([plot_df, user_point], ignore_index=True)

            color_scale = alt.Scale(domain=["Normal", "Anomaly", "Your Input"],
                                    range=["steelblue", "crimson", "gold"])

            scatter = alt.Chart(plot_df).mark_circle(size=80).encode(
                x="x:Q",
                y="y:Q",
                color=alt.Color("label:N", scale=color_scale),
                tooltip=["label"]
            ).properties(width=700, height=500).interactive()

            st.altair_chart(scatter, use_container_width=True)
        else:
            st.warning("Please enter some text to analyze.")


    # Visuals after demo
    with st.expander("🔠 Word Cloud"):
        text_corpus = " ".join(data_train['clean_text'].dropna())
        st.pyplot(create_word_cloud(text_corpus))

    with st.expander("😊 Sentiment Analysis"):
        sentiment_scores = data_train['clean_text'].apply(lambda x: sia.polarity_scores(x))
        sentiment_df = pd.DataFrame(list(sentiment_scores))
        st.altair_chart(create_sentiment_chart(sentiment_df), use_container_width=True)

    with st.expander("📚 Text Complexity"):
        data_train['flesch'] = data_train['clean_text'].apply(flesch_reading_ease)
        data_train['gunning_fog'] = data_train['clean_text'].apply(gunning_fog)
        col1, col2 = st.columns(2)
        col1.metric("Avg Readability Score", f"{data_train['flesch'].mean():.1f}")
        col2.metric("Gunning Fog Index", f"{data_train['gunning_fog'].mean():.1f}")

def render():
    main()
