import streamlit as st
from tabs import st_cluster_combo, sentiment_analysis, LDA_model



st.set_page_config(
    layout="wide",
    page_title="NLP Insights Dashboard",
)
st.title("Multi-File Streamlit Dashboard")

tab_options = {
    "Sentiment Analysis": sentiment_analysis.render,
    "LDA Model": LDA_model.render,
    "Cluster Bert": st_cluster_combo.render
}
selected_tab = st.sidebar.radio("Choose a dashboard tab", list(tab_options.keys()))

# Only render the selected tab
tab_options[selected_tab]()

# tab1, tab2, tab3 = st.tabs(["Sentiment Analysis","LDA Model","Cluster Bert" ])


# with tab1:
       
#     sentiment_analysis.render()

# with tab2:
   
#     LDA_model.render()

# with tab3:

#     st_cluster_combo.render()



