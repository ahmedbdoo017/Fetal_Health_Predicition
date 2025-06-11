import streamlit as st
import pickle
import numpy as np

# Load your model and scaler
model = pickle.load(open(r'GBC_Semi_final_model.pkl', 'rb'))
scaler = pickle.load(open(r'GBC_Semi_final_model_scaler.pkl', 'rb'))

def predict(features):
    # Preprocess features
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return prediction

def main():
    st.title('Fetal Health Predicition')

    # Add input fields for all 19 features
    feature1 = st.number_input('Feature 1', min_value=0.0, max_value=100.0, value=0.0)
    feature2 = st.number_input('Feature 2', min_value=0.0, max_value=100.0, value=0.0)
    feature3 = st.number_input('Feature 3', min_value=0.0, max_value=100.0, value=0.0)
    feature4 = st.number_input('Feature 4', min_value=0.0, max_value=100.0, value=0.0)
    feature5 = st.number_input('Feature 5', min_value=0.0, max_value=100.0, value=0.0)


    if st.button('Predict'):
        features = [feature1, feature2, feature3, feature4, feature5]
        prediction = predict(features)
        st.write(f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    main()