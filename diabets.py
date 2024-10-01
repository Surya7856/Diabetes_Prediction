import streamlit as st
import numpy as np
import pickle

file1=open('diab.pkl','rb')
rf=pickle.load(file1)
file1.close()

st.set_page_config(page_title="Diabets prediction")

st.title('Diabetes Predictor')

Pregnancies=st.number_input('Number of Pregnancies')

Glucose=st.number_input('Glucose level(0-200):')

BloodPressure=st.number_input('BloodPressure(0-120):')

SkinThickness=st.number_input('SkinThickness in (0-100):')

Insulin=st.number_input('Insulin (0-800)')

BMI=st.number_input('BMI')

DiabetesPedigreeFunction=st.number_input('DiabetesPedigreeFunction (0.0 to 2.50):')

Age=st.number_input('Age')

if st.button('Check'):
    query = np.array([Pregnancies, Glucose, BloodPressure,SkinThickness,Insulin,
                      BMI, DiabetesPedigreeFunction, Age])
    query=query.reshape(1,8)
    probabilities = rf.predict_proba(query)
    print(probabilities)
    
    # Get probability of positive class (class 1)
    positive_class_prob = probabilities[:, 1]
    st.title("You have a "+str(np.round(positive_class_prob*100,2))+"% chance of having diabetes.\nConsult a doctor for further evaluation.")

st.sidebar.title("About")
st.sidebar.write("""Welcome to DiabetesPredict!\n
Take control of your health with our innovative diabetes risk assessment tool.\n
Predict. Prevent. Thrive.\n
Simply input your health data\n
Get started now and take the first step towards a diabetes-free life!""")



                        





