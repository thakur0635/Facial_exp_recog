import time
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Facial Expression Recognition", page_icon='face-detection.png')





emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']



st.title('Facial Expression Recognition')
label=''
scaled_predictions=[]
with st.container():
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg'], accept_multiple_files=False)

if st.button('Recognize Expression', type='primary'):
    if uploaded_file is None:
         st.error('Please upload a image')
    else: 
        with st.spinner('Please Wait...'):
            face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
            classifier =load_model(r'model.h5')
            with open("temp.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())
            file_path = "temp.jpg"
            frame = cv2.imread(file_path)
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)
            print(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    normalized_prediction = np.array(prediction) / np.sum(prediction)
                    scaled_prediction = normalized_prediction * 10
                    print(label,prediction,scaled_predictions)
                else:
                    print('No label')

        with st.container():
            # with st.chat_message("user"):
            st.success('Expression Recognized Successfully!', icon="✅")
            st.write("Expression ➡️ ",label)
            labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            fig = plt.bar(labels, prediction, color='blue')
            plt.xlabel('Emotion')
            plt.ylabel('Prediction')
            plt.title('Prediction for Each Emotion')
            plt.savefig('prediction_chart.png')
            st.image('prediction_chart.png', width=500)
            # col1, col2, col3,col4,col5,col6,col7 = st.columns(7)
            # with col1:
            #     a = st.slider('Angry', min_value=0.0, max_value=1.0, value=scaled_predictions[0],disabled=True)
            # with col2:
            #     b = st.slider('Disgust', min_value=0.0, max_value=1.0, value=scaled_predictions[1],disabled=True)
            # with col3:
            #     c = st.slider('Fear', min_value=0.0, max_value=1.0, value=scaled_predictions[2],disabled=True)
            # with col4:
            #     d = st.slider('Happy', min_value=0.0, max_value=1.0, value=scaled_predictions[3],disabled=True)
            # with col5:
            #     e = st.slider('Neutral', min_value=0.0, max_value=1.0, value=scaled_predictions[4],disabled=True)
            # with col6:
            #     f = st.slider('sad', min_value=0.0, max_value=1.0, value=scaled_predictions[5],disabled=True)
            # with col7:
            #     g = st.slider('Surprise', min_value=0.0, max_value=1.0, value=scaled_predictions[6],disabled=True)
