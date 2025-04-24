import streamlit as st
import pandas as pd 
import numpy as np 
import ast
import plotly.express as px
import pickle
import plotly.graph_objects as go

# add data from other runs, make sure plots are interactive 

# Page configuration
# st.set_page_config(page_title="Cluster Analysis Dashboard", layout="wide")
def render():
    st.header('Unsupervised Bert/Kmeans classification :bar_chart:')
    st.write("""
    Give me an example !

    """)

    # Data loading with caching
    @st.cache_data
    def load_data():
        try:
            vec_data_3D = np.load("tabs/TSNE_3.npy")
            vec_data_2D = np.load("tabs/TSNE.npy")
            train = pd.read_csv('tabs/trainSA.csv')
            test = pd.read_csv('tabs/testSA.csv')
            
            text_data = pd.concat([train, test]).reset_index(drop=True)
            bert_emb=np.load("tabs/embedding.npy")
            # Load and combine optimization results
            
            return vec_data_3D, vec_data_2D, text_data,bert_emb
   
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None, None, None
    @st.cache_data
    def load_labels():
        topics_label=np.load('tabs/bert_model_regular_labels.npy')
        topics=np.load('tabs/label_topics.npy',allow_pickle=True)
        anomalies= np.load('tabs/anomaly_labels.npy',allow_pickle=True)
        return topics_label,topics, anomalies
        
    @st.cache_data
    def load_models():
        # load bert embedding 
        # with open("tabs/sb_model.pkl", "rb") as f:
        #     sb_model = pickle.load(f)
        from sentence_transformers import SentenceTransformer
        sb_model = SentenceTransformer("all-MiniLM-L6-v2")

        # load TSNE model 
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=3)
        
        # loading kmean model
        with open("tabs/kmean_model.pkl", "rb") as f:
            kmean_model = pickle.load(f)
        
        return sb_model,tsne, kmean_model

# reading in data and labels and models 
    v_data_3D, v_data_2D, text_data,bert_emb = load_data()
    bert_model , tsne_model,kmean_model = load_models()
    topics_labels,topics, anomaly_labels= load_labels()
    

    # making dataframe with all the info to be plotted later
#account:receive,money:come,seller:return.
    plt_data = pd.DataFrame(v_data_3D, columns= ['t-SNE1','t-SNE2','t-SNE3'])
    # labels
    plt_data['labels']=topics_labels
    # text 
    plt_data['topics']=topics
    # actual topics
    plt_data['text']=text_data['text']
    # anomlies
    plt_data['anomaly']=anomaly_labels
    # number of documnets per topic 
    dic={}
    for label in np.unique(plt_data['labels']):
        text=plt_data['text'][plt_data['labels']==label]
        dic[label]=(len(text)) # making list of total datapoints for each class
    plt_data['topic_count']= plt_data['labels'].apply(lambda x: dic[x])

   
    # text prompt for interaction
    user_input = st.text_area("Enter a sentence or paragraph:", height=150)
    viz=st.checkbox('visual of prediction')

    if st.button("Detect"):
    # if user_input: # if prompot 
        # getting predictions 
        with st.spinner("Making predictions"):
            # with TSNE u can't make a single prediction so will add embedding of text to the 
            # entire file. Then I'll make get the lab of the last cluster and make preictions
            
            ui_embeddings = bert_model.encode(user_input)
            prediction=kmean_model.predict(ui_embeddings.reshape(1, -1))[0] # getting the value
            topic=plt_data['topics'][plt_data['labels']==prediction].iloc[0]
            st.write(prediction)
            st.write(plt_data[plt_data['labels']==prediction].head())
            anomaly_level=plt_data['anomaly'][plt_data['labels']==prediction].iloc[0]
            st.write(f'Naturally, text falls under {topic}.')
            st.write(f'Level of importance is {anomaly_level}.')

        with st.spinner("Making visuals"):
            if viz:
                stacked = np.vstack((bert_emb, ui_embeddings))
                st.write('getting cords')
                ui_cords = tsne_model.fit_transform(stacked)[-1]
                st.write('got cords')
                st.write('getting predictions')
                st.header('Plotting 3D of anomaly levels', divider=True)
                fig2=px.scatter_3d(plt_data,x='t-SNE1',y='t-SNE2',z='t-SNE3', color='anomaly'
                            , opacity=0.7,hover_data=['text','topic_count','topics'],
                                title=f"Kmeans clustering with TSNE reduction Anomalies")

                fig2.update_traces(marker=dict(size=3))

            # plot of prediction 

                fig2.add_trace(
        go.Scatter3d(
            x=[ui_cords[0]],  # Wrap in a list to ensure correct shape
            y=[ui_cords[1]],
            z=[ui_cords[2]],
            mode='markers',
            marker=dict(
                color='red',
                size=4,  # Make it larger than other points
                symbol='x'  # Use a distinct symbol
            ),
            name="The InPut",  # Legend label
            hoverinfo='text',
            hovertext=f"Text: {user_input}<br>Topic: {topic}<br>Anomaly Level: {anomaly_level}" # Show the user's text on hover
        )
    )

                st.plotly_chart(fig2)
            
            # making visual if viz not selected 
            else:
                with st.spinner("Making visuals"):
                    st.header('Plotting 3D of all topics', divider=True)
                    plt_data=plt_data.sort_values(by='topics')
                    fig=px.scatter_3d(plt_data,x='t-SNE1',y='t-SNE2',z='t-SNE3', color='topics'
                                , opacity=0.7,hover_data=['text','topic_count','topics'],
                                    title=f"Kmeans clustering with TSNE reduction Topics")

                    fig.update_traces(marker=dict(size=3))
                    st.plotly_chart(fig)
                    
                    # Creating plot for anomalies 
                    st.header('Plotting 3D of anomaly levels', divider=True)
                    fig2=px.scatter_3d(plt_data,x='t-SNE1',y='t-SNE2',z='t-SNE3', color='anomaly'
                                , opacity=0.7,hover_data=['text','topic_count','topics'],
                                    title=f"Kmeans clustering with TSNE reduction Anomalies")

                    fig2.update_traces(marker=dict(size=3))
                    st.plotly_chart(fig2)
      


    else: # Displaying regular topics and anomalies if no prompt entered
         # Create interactive 3D plot for topics 
        with st.spinner("Making visuals"):
            st.header('Plotting 3D of all topics', divider=True)
            plt_data=plt_data.sort_values(by='topics')
            fig=px.scatter_3d(plt_data,x='t-SNE1',y='t-SNE2',z='t-SNE3', color='topics'
                        , opacity=0.7,hover_data=['text','topic_count','topics'],
                            title=f"Kmeans clustering with TSNE reduction Topics")

            fig.update_traces(marker=dict(size=3))
            st.plotly_chart(fig)
            
            # Creating plot for anomalies 
            st.header('Plotting 3D of anomaly levels', divider=True)
            fig2=px.scatter_3d(plt_data,x='t-SNE1',y='t-SNE2',z='t-SNE3', color='anomaly'
                        , opacity=0.7,hover_data=['text','topic_count','topics'],
                            title=f"Kmeans clustering with TSNE reduction Anomalies")

            fig2.update_traces(marker=dict(size=3))
            st.plotly_chart(fig2)
