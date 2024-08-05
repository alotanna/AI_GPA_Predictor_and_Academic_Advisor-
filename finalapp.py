import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import tensorflow as tf
import joblib
import openai
import base64

# Load the trained model, scaler, and encoder
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('encoder.pkl')
except Exception as e:
    st.error(f"Error loading model, scaler, or encoder: {str(e)}")
    st.stop()


# Set your OpenAI API key
openai.api_key = "sk-proj-nx2fCMkFkbR0UVwlYoCMT3BlbkFJ4eiMK47mcR6PI460gI17"  # Replace with your actual API key

# Streamlit app
st.set_page_config(page_title="Student GPA Predictor and Advisor", page_icon=":trophy:")

# Function to get advice using OpenAI's GPT-3.5
def get_advice(predicted_gpa, study_hours, stress_level, time_wasted_on_social_media, previous_gpa,
               physical_activity, educational_resources, nutrition, sleep_patterns, name):
    prompt = f"""
    Based on the following student information for {name}, provide personalized academic advice:

    Predicted GPA: {predicted_gpa:.2f}
    Study Hours per Week: {study_hours}
    Stress Level: {stress_level}
    Time Wasted on Social Media: {time_wasted_on_social_media} hours per day
    Previous GPA: {previous_gpa:.2f}
    Physical Activity: {physical_activity} hours per week
    Educational Resources Availability: {educational_resources}
    Nutrition: {nutrition}
    Sleep Patterns: {sleep_patterns} hours per day

    Please provide specific, actionable advice to help {name} improve their academic performance and overall well-being. You can also tell them where they are going wrong.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are an experienced academic advisor, skilled in providing personalized advice to students based on their academic performance and lifestyle factors."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        color: #333;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 4px;
        padding: 8px 16px;
        border: none;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .stTextInput > div > div > input, .stSelectbox > div > div > select {
        border-radius: 4px;
        border: 1px solid #ced4da;
        padding: 8px;
    }
    .home-container {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 30px;
        border-radius: 10px;
    }
    .content-container {
        background-color: white;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #007bff;
    }
    </style>
""", unsafe_allow_html=True)

# Home page
def home_page():
    add_bg_from_local('webpics.jpg')  # Make sure to replace 'webpics.jpg' with your actual background image file
    st.markdown("""
    <div class="home-container">
        <h1>Welcome to Student Success Predictor</h1>
        <p style="font-size: 18px; margin-bottom: 30px;">
            Unlock your academic potential and pave the way to success with our advanced GPA prediction and personalized advice system.
        </p>
        <h2>Why Use Our Predictor?</h2>
        <ul style="text-align: left; max-width: 600px; margin: 0 auto; padding-left: 20px;">
            <li>Accurate GPA predictions based on multiple factors</li>
            <li>Personalized advice from AI-powered academic advisors</li>
            <li>Insights into areas for improvement</li>
            <li>Track your progress over time</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button('Get Started'):
        st.session_state.page = 'predict'

def predict_page():
    st.title('Student GPA Predictor and Advisor')

    st.header('Enter Your Information')

    # Input for user's name
    name = st.text_input("Enter your name", key="user_name")

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="content-container">', unsafe_allow_html=True)

        # Input fields for column 1
        study_hours = st.number_input('Study Hours per Week', 0, 80, 20, key='study_hours')
        physical_activity = st.number_input('Physical Activity (hours per week)', 0, 100, 1, key='physical_activity')
        time_wasted_on_social_media = st.number_input('Time Wasted on Social Media (hours per day)', 0, 24, 1,
                                                      key='social_media')
        sleep_patterns = st.number_input('Sleep time (hours per day)', min_value=0, max_value=24, step=1,
                                         key='sleep_patterns')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        # Input fields for column 2
        previous_gpa = st.number_input("Previous GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1,
                                       key='previous_gpa')
        stress_level = st.selectbox('Stress Level', ['Low', 'Medium', 'High'], key='stress_level')
        educational_resources = st.selectbox('Educational Resources Availability', ['Low', 'Medium', 'High'],
                                             key='educational_resources')
        nutrition = st.selectbox('Nutrition', ['Low', 'Medium', 'High'], key='nutrition')
        st.markdown('</div>', unsafe_allow_html=True)

    # Create a new section below the columns for the progress bar and buttons
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    col3, _ = st.columns([2, 1])  # Only one column needed here

    with col3:
        if st.button('Predict GPA'):
            # Create a DataFrame with the input data
            input_dict = {
                'Time_Wasted_on_Social_Media': time_wasted_on_social_media,
                'Previous GPA': previous_gpa,
                'Study_Hours': study_hours,
                'Physical_Activity': physical_activity,
                'Stress_Levels': stress_level,
                'Educational_Resources': educational_resources,
                'Nutrition': nutrition,
                'Sleep_Patterns': sleep_patterns
            }

            df_input = pd.DataFrame([input_dict])

            # Transform categorical features
            categorical_features = ['Nutrition', 'Stress_Levels', 'Educational_Resources']

            for col in categorical_features:
                df_input[col] = label_encoder.fit_transform(df_input[col])

            # Scale the input features
            scaled_input = scaler.transform(df_input)

            # Make prediction
            prediction = model.predict(scaled_input)
            predicted_gpa = prediction[0][0]

            st.session_state.predicted_gpa = predicted_gpa
            st.session_state.input_data = input_dict  # Save the input data for use in advice

            # Display the predicted GPA
            st.session_state.show_gpa = True

            # Create a smaller circular progress bar
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_gpa,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "GPA Scale", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#007bff"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 4], 'color': '#e6f2ff'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 3.0}}))

            fig.update_layout(
                autosize=False,
                width=150,  # Smaller width
                height=150,  # Smaller height
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor="white",
                font={'color': "darkblue", 'family': "Arial"}
            )

            st.session_state.gpa_chart = fig

        if 'show_gpa' in st.session_state and st.session_state.show_gpa:
            st.write(f"Predicted GPA: {st.session_state.predicted_gpa:.2f}")
            st.plotly_chart(st.session_state.gpa_chart, use_container_width=True)

        if st.button('Get Advice'):
            if 'predicted_gpa' in st.session_state:
                # Get advice from OpenAI with all inputs and predicted GPA
                advice = get_advice(
                    st.session_state.predicted_gpa,
                    study_hours,
                    stress_level,
                    time_wasted_on_social_media,
                    previous_gpa,
                    physical_activity,
                    educational_resources,
                    nutrition,
                    sleep_patterns,
                    name if name else "the student"
                )
                st.write("### Personalized Advice:")
                st.write(advice)
            else:
                st.error("Please predict the GPA first before getting advice.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Add a back button to return to the home page
    if st.button('Back to Home'):
        st.session_state.page = 'home'

# Main app logic
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'predict':
    predict_page()
