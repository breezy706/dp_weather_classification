import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import base64
import cv2
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather Prediction", layout="centered")

# ======== FUNCTION TO ADD BACKGROUND IMAGE ========
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call it here (use your image path)
add_bg_from_local("C://Users//Admin//OneDrive//Desktop//weather//Weather//Technology.jpg")


st.markdown("<h1 style='color: black;'>DATA SCIENTIST: BREEZY_YRN</h1>", unsafe_allow_html=True)
img_size = 100
PRETRAINED_MODEL_PATH = "C://Users//Admin//OneDrive//Desktop//weather//Weather//weather_Trained_mode.h5"
CATEGORIES = ["Cloudy", "Rain" ,"Shine","Sunrise"]

model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
print('Model Loaded')


def prepare(image):
    img_array = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array, img_array  # return resized for prediction & original for display

def load_classifier():
    st.markdown("<h3 style='font-weight: bold;color: black;'>UPLOAD WEATHER IMAGE FOR PREDICTION</h3>",unsafe_allow_html=True)
    file = st.file_uploader(" ", type=['jpg', 'jpeg', 'png'])

    if file is not None:
        resized_img, display_img = prepare(file)
        image_array_expanded = np.expand_dims(resized_img / 255.0, axis=0)

        st.image(display_img, caption="Uploaded Image", use_column_width=True)
        st.write("")

        if st.button("PREDICT"):
            # Make prediction
            prediction = model.predict(image_array_expanded)[0]
            predicted_class_idx = np.argmax(prediction)
            predicted_class = CATEGORIES[predicted_class_idx]

            # Display predicted class
            st.markdown(f"### <span style='color:black;'>Predicted Weather: <b>{predicted_class}</b></span>", unsafe_allow_html=True)


            # Display bar chart
            st.markdown(
    "<h3 style='color: black;'>MORE INFORMATION OF THE IMAGE UPLOADED</h3>",
    unsafe_allow_html=True
)
            fig, ax = plt.subplots()
            ax.bar(CATEGORIES, prediction)
            ax.set_xlabel("Weather")
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)

            # Display formatted probabilities with 9 decimal places, in two lines
            formatted_probs = [f"{p:.9f}" for p in prediction]

            #st.subheader("Detailed Probabilities:")
            
            # Add black color + spacing between class items (using margin-right)
            half = len(CATEGORIES) // 2
            line1 = " ".join([
                f"<span style='color:black; margin-right: 40px;'>{cat}: {prob}</span>"
                for cat, prob in zip(CATEGORIES[:half], formatted_probs[:half])
])
            line2 = " ".join([
                f"<span style='color:black; margin-right: 40px;'>{cat}: {prob}</span>"
                for cat, prob in zip(CATEGORIES[half:], formatted_probs[half:])
])
# Display in Streamlit
            st.markdown(
     "<h4 style='font-weight: bold;color: black;'>Detailed Probabilities:</h4>",
    unsafe_allow_html=True
)
            st.markdown(line1, unsafe_allow_html=True)
            st.markdown(line2, unsafe_allow_html=True)

def main():
    load_classifier()


if __name__ == "__main__":
	main()

    


    