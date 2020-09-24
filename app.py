
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image,ImageOps

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/content/drive/My Drive/COVID19 Detection/model_cov.h5')
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model
model = load_model()

st.title('Covid19 Test Using the Chest X-ray')
st.header("This application is built using ML & DL Algorithm.")

file = st.file_uploader('Upload your chest X-ray',type=['jpg','jpeg','png'])
# file = cv2.imread(file)
# st.success(file)


def importAndPredict(img,model):
  # img = img.getdata()
  # img = np.array(img)
  # img = img.astype('uint8')
  img = cv2.resize(np.float32(img),(224,224))
  img = np.reshape(img,(1,224,224,3))
  classes = int(model.predict(img))
  label = ["COVID INFECTED","NORMAL"]
  # covid.append(classes)
  return label[classes]

if file is None:
  st.text('Please upload your image first.')
else:
  img = Image.open(file)
  # st.success(img)
  # img = cv2.imread(file)
  # img = cv2.resize(np.float32(img),(224,224))
  st.image(img,use_column_width=True)
 
  prediction = importAndPredict(img,model)
  st.success(prediction)