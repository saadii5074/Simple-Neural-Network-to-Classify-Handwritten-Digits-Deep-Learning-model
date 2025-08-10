# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageOps
import os

MODEL_FILE = "mnist_model.h5"

# Train model if not exists
if not os.path.exists(MODEL_FILE):
    st.write("Training model... Please wait.")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save(MODEL_FILE)
else:
    model = load_model(MODEL_FILE)

st.title("🖊 Handwritten Digit Recognition")
st.write("Upload your digit image and the model will predict it!")

uploaded_file = st.file_uploader("Upload an image (digit)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    image = ImageOps.invert(image)  # Invert colors if background is white
    image = image.resize((28, 28))
    
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)
    
    st.image(image.resize((200, 200)), caption="Processed Image", use_column_width=False)
    
    prediction = model.predict(img_array)
    st.write(f"### Predicted Digit: **{np.argmax(prediction)}**")
    st.bar_chart(prediction[0])
