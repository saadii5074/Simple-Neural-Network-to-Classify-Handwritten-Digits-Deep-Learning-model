import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 3. Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 4. Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5. Streamlit UI
st.title("Handwritten Digit Recognition (Custom Image Test)")

# Predefined image path
image_path = "C:/Users/isaad/Desktop/image.JPG.avif"

try:
    # Read and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # invert colors if needed
    img = img / 255.0
    img = img.reshape(1, 28, 28)

    # Prediction
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)

    st.image(image_path, caption="Uploaded Handwritten Digit", use_column_width=True)
    st.write(f"Predicted Digit: **{predicted_label}**")

except Exception as e:
    st.error(f"Error reading the image: {e}")
