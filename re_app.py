import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

### Load the trained model...
model = tf.keras.models.load_model("regression_model.h5")

### load encoder and scaler...
with open('re_label_encoder_gender.pkl', 'rb') as file:
    re_label_encoder_gender = pickle.load(file)

with open('re_onehot_encoder_geo.pkl', 'rb') as file:
    re_label_encoder_geo = pickle.load(file)    
    
with open('re_scaler.pkl', 'rb') as file:
    re_scaler = pickle.load(file)


## Streamlit app...
st.title('Customer Salary Prediction')

# User Input...

gender = st.selectbox('Gender', re_label_encoder_gender.classes_)
geography = st.selectbox('Geography', re_label_encoder_geo.categories_[0])
age = st.slider('Age', min_value=18, max_value=92)
tenure = st.slider('Tenure', min_value=0, max_value=10)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.number_input('Exited')

# Prepare the input data...
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [re_label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

geography_encoded = re_label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geography_encoded, columns=re_label_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data...
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data...
input_data_scaled = re_scaler.transform(input_data)

# Predict Churn...
prediction = model.predict(input_data_scaled)
prediction_ = prediction[0][0]

#st.write(f"The customer is likey churn")
st.write(f"EstimatedSalary for this client: ${prediction_}") 


