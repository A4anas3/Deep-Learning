import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ===============================
# Load trained model and encoders
# ===============================
try:
    model = tf.keras.models.load_model('churn_model.keras')  # new format (recommended)
except:
    model = tf.keras.models.load_model('churn_model.h5', compile=False)  # fallback

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehotencoder = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    labelencoder = pickle.load(f)

# ===============================
# Streamlit UI
# ===============================
st.title("💼 Customer Churn Prediction App")

# Collect user input
geography = st.selectbox('🌍 Geography', onehotencoder.categories_[0])
gender = st.selectbox('👤 Gender', labelencoder.classes_)
age = st.slider('🎂 Age', 18, 92)
balance = st.number_input('💰 Balance', min_value=0.0, step=100.0)
credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, step=1)
estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, step=100.0)
tenure = st.slider('⏳ Tenure (years)', 0, 10)
num_of_products = st.slider('🛍 Number of Products', 1, 4)
has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

# =================================
# Predict only on button click
# =================================
if st.button("🔍 Predict Churn"):
    # Encode categorical variables
    gender_encoded = labelencoder.transform([gender])[0]
    geo_encoded = onehotencoder.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehotencoder.get_feature_names_out(['Geography'])
    )

    # Create input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Combine with encoded geography
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    y = model.predict(input_scaled)
    y_prob = y[0][0]

    # Result
    if y_prob > 0.5:
        st.error(f"⚠️ The customer is **likely to churn** (Probability: {y_prob:.2f})")
    else:
        st.success(f"✅ The customer is **unlikely to churn** (Probability: {y_prob:.2f})")
