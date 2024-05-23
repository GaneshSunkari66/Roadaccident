import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# try:
#     model = pickle.load(open('selected_features.pkl', 'rb'))
# except Exception as e:
#     st.error(f"Error loading the model: {e}")
# Load the trained model
model = pickle.load(open('features.pkl', 'rb'))

def predict_accident(Sex_of_driver, Type_of_vehicle, Types_of_Junction, Road_surface_conditions, Type_of_collision, Casualty_class, Casualty_severity,Cause_of_accident,Driving_experience_encoded):
    # Create input array
    features = np.array([Sex_of_driver, Type_of_vehicle, Types_of_Junction, Road_surface_conditions, Type_of_collision, Casualty_class, Casualty_severity,Cause_of_accident,Driving_experience_encoded]).reshape(1, -1)

    # Predict
    prediction = model.predict(features)

    return prediction[0]

def main():
    st.title('Road Accident Prediction')
    st.write('Fill out the form below to predict the Accident.')

    # Input form
    with st.form(key='accident_form'):
        Sex_of_driver = st.radio('Sex_of_driver', ['male','female'])
        Type_of_vehicle=st.selectbox('Type_of_vehicle',['Automobile','Public (> 45 seats)','Lorry (41?100Q)','Long lorry','Lorry (11?40Q)','Public (13?45 seats)','Public (12 seats)','Taxi','Pick up upto 10Q','Stationwagen','Ridden horse','Other','Bajaj','Turbo','Motorcycle'])
        Types_of_Junction = st.selectbox('Types_of_Junction', ['No junction','Y Shape','Crossing','O Shape','Other','Unknown'])
        Type_of_collision = st.selectbox('Type_of_collision',['Collision with roadside-parked vehicles','Vehicle with vehicle collision','Collision with roadside objects','Collision with animals','Rollover','Fall from vehicles','Collision with pedestrians','Other'])
        Casualty_class = st.selectbox('Casualty_class',['Driver or rider','Pedestrian','Passenger','na'])
        Casualty_severity = st.number_input('Casualty_severity', min_value=0, max_value=3, step=1)
        Road_surface_conditions = st.selectbox('Road_surface_conditions', ['Dry','Wet or damp','Snow'])
        Cause_of_accident=st.selectbox('Cause_of_accident',['Moving Backward','Overtaking','Changing lane to the left','Changing lane to the right','Overloading','No priority to vehicle','No priority to pedestrian','No distancing','Getting off the vehicle improperly','Improper parking','Other'])
        Driving_experience_encoded = st.number_input('Driving_experience_encoded', min_value=0, max_value=3, step=1)

        submit_button = st.form_submit_button(label='Predict')

       # Convert categorical features to numeric
    Type_of_vehicle = 1 if Type_of_vehicle == ['Automobile','Public (> 45 seats)','Lorry (41?100Q)','Long lorry','Lorry (11?40Q)''Public (13?45 seats)','Public (12 seats)'] else 0
    Types_of_Junction = 1 if Types_of_Junction == ['Y Shape','Crossing','O Shape'] else 0
    Type_of_collision = 1 if Type_of_collision == ['Collision with roadside-parked vehicles','Vehicle with vehicle collision','Collision with roadside objects'] else 0
    Sex_of_driver = 1 if Sex_of_driver == 'Male' else 0
    Casualty_class = 1 if Casualty_class == ['Driver or rider','Pedestrian','Passenger'] else 0
    Road_surface_conditions = 1 if Road_surface_conditions == 'Snow' else 0
    Cause_of_accident=1 if Cause_of_accident==['Moving Backward','Overtaking','Changing lane to the left','Changing lane to the right','Overloading'] else 0

    # Perform prediction when form is submitted
    if submit_button:
        prediction = predict_accident(Sex_of_driver, Type_of_vehicle, Types_of_Junction, Road_surface_conditions, Type_of_collision, Casualty_class, Casualty_severity,Cause_of_accident,Driving_experience_encoded)
        if prediction == 1:
            st.error('Serious Injury')
        else:
            st.success('Slight Injury')

if __name__ == '__main__':
    main()

