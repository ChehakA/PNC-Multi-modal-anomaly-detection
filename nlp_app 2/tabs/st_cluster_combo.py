import streamlit as st
import pandas as pd 
import numpy as np 
import ast
import plotly.express as px

# add data from other runs, make sure plots are interactive 

# Page configuration
# st.set_page_config(page_title="Cluster Analysis Dashboard", layout="wide")
def render():
    st.header('Parameter Distribution for Cluster Labels :bar_chart:')
    st.write("""
    - **min_cluster_size**: [5,10,25,50,75,100,125,150,175,200,225,250,275,300]
    - **min_samples**: [5,10,25,50,75,100,125,150,175,200]
    - **cluster_selection_method**: ['leaf','eom']
    """)
    st.write("*Note: Missing combinations indicate clusters with <3 or >40 labels*")
    # Data loading with caching
    @st.cache_data
    def load_data():
        try:
            vec_data_3D = np.load("tabs/TSNE_3.npy")
            vec_data_2D = np.load("tabs/TSNE.npy")
            train = pd.read_csv('tabs/trainSA.csv')
            test = pd.read_csv('tabs/testSA.csv')
            
            text_data = pd.concat([train, test]).reset_index(drop=True)
            
            # Load and combine optimization results
            parts = []
            for i in range(4, 10):
                df = pd.read_csv(f'tabs/hdbscan_optimization_results_{i}.csv')
                params_df = pd.json_normalize(df['params'].apply(ast.literal_eval))
                parts.append(pd.concat([params_df, df[['labels', 'score', 'prop_score']]], axis=1))
                
            feat_df = pd.concat(parts, axis=0)
            return vec_data_3D, vec_data_2D, text_data, feat_df
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None, None, None

    v_data_3D, v_data_2D, text_data, feat_df = load_data()
    
    st.sidebar.header("For Bert CLustering!!")
    st.sidebar.write("Clustering Parameters HDBSCAN")
    min_samples = st.sidebar.select_slider("Minimum samples", options=np.unique(feat_df['min_samples']))
    min_cluster_size = st.sidebar.select_slider("Minimum cluster size", options=np.unique(feat_df['min_cluster_size']))
    cluster_method = st.sidebar.select_slider("Cluster method", options=np.unique(feat_df['cluster_selection_method']))

    # Data filtering
    filter_mask = (
        (feat_df['min_samples'] == min_samples) & 
        (feat_df['min_cluster_size'] == min_cluster_size) & 
        (feat_df['cluster_selection_method'] == cluster_method)
    )

    labels = feat_df.loc[filter_mask, 'labels']
    prop_score = feat_df.loc[filter_mask, 'prop_score']

    # Handle missing data
    if labels.empty:
        st.warning("Selected combination doesn't exist in the data")
    else:
            
        labels = ast.literal_eval(labels.iloc[0])
        prop_score = ast.literal_eval(prop_score.iloc[0]) if not prop_score.empty else []

        


        # making daata frame to plot all info easily
        plt_data = pd.DataFrame(v_data_3D, columns= ['t-SNE1','t-SNE2','t-SNE3'])
        plt_data['labels']=labels
        plt_data['text']=text_data['text']
        plt_data['score']=prop_score
        plt_data['t-SNE1_2d']=v_data_2D[:,0]
        plt_data['t-SNE2_2d']=v_data_2D[:,1]


        #Creating 2D plot 
        st.header('Plotting 2D :two:',divider=True)
        fig_2=px.scatter(plt_data,x='t-SNE1_2d',y='t-SNE2_2d', color='labels', color_continuous_scale='viridis'
                    , opacity=1,hover_data=['labels','text','score'],
                        title=f"HBDSCAN Clustering (min_samp: {min_samples}, min_clus: {min_cluster_size}, type: {cluster_method})")

        fig_2.update_layout(
            xaxis_title="Feature 1",
            yaxis_title="Feature 2"
            )

        st.plotly_chart(fig_2)

        # Create interactive 3D plot
        st.header('Plotting 3D :three:', divider=True)
        fig=px.scatter_3d(plt_data,x='t-SNE1',y='t-SNE2',z='t-SNE3', color='labels', color_continuous_scale='viridis'
                    , opacity=1,hover_data=['labels','text','score'],
                        title=f"HBDSCAN Clustering (min_samp: {min_samples}, min_clus: {min_cluster_size}, type: {cluster_method})")

        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            xaxis_title="Feature 1",
            yaxis_title="Feature 2"
            
        )
        st.plotly_chart(fig)
