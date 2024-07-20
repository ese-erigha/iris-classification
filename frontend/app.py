import joblib
import streamlit as st
import numpy as np
from sklearn.datasets import load_iris

@st.cache_resource
def load_model():
    model_path = 'model.joblib'
    if(st.secrets["prod"] == True):
        model_path = "./frontend/model.joblib"

    model = joblib.load(model_path)
    return model


@st.cache_data
def load_category():
    dataset = load_iris()
    return dataset.target_names


st.title('Iris Species Prediction')

sepal_length = st.slider('Sepal Length', 4.0, 8.0)
sepal_width = st.slider('Sepal Width', 2.0, 5.0)
petal_length = st.slider('Petal Length', 1.0, 6.9)
petal_width = st.slider('Petal Width', 0.0, 2.5)

# Convert to float
sepal_length=float(sepal_length)
sepal_width=float(sepal_width)
petal_length = float(petal_length)
petal_width = float(petal_width)

if st.button('Predict'):
    input = [sepal_length, sepal_width, petal_length,petal_width]
    
    # Perform prediction 
    features = np.array(input).reshape(1, -1)
    model = load_model()
    prediction = model.predict(features)
    classes = load_category()
    class_name = classes[prediction][0].capitalize()
    st.html("<p></p>")
    st.subheader(f'Predicted class is :blue[{class_name}]')
