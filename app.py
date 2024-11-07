# Contents of ~/my_app/streamlit_app.py
import streamlit as st
import base64
import streamlit as st
import pickle

# st.title("Emotion and Speech Recognition")

st.markdown(f'<h1 style="color:#FF0000;font-size:45px;">{"Emotion and Speech Recognition"}</h1>', unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('4.jpg') 

