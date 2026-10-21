import pandas as pd
import numpy as np  
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def load_data():
    """Cache the iris dataset loading."""
    iris = load_iris()  
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])
    df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

@st.cache_resource
def train_model():
    """Cache the trained model."""
    df = load_data()
    model = LogisticRegression(max_iter=200)
    model.fit(df.iloc[:,:-1], df['target'])
    return model

def main():
    # Load cached data and model
    df = load_data()
    model = train_model()
    model_accuracy = model.score(df.iloc[:,:-1], df['target']) * 100
    
    # Rest of your app code remains the same...
    st.title("Iris Dataset Scatter Plot with Logistic Regression Predictions")
    # ... [rest of your code]

if __name__ == "__main__":
    main()