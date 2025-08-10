import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Load Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2️⃣ Normalize (0-255 -> 0-1 range)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3️⃣ Build Model
model = Sequential([
    Flatten(input_shape=(28, 28)),     # Flatten 28x28 image -> 784 vector
    Dense(128, activation='relu'),     # Hidden layer
    Dense(64, activation='relu'),      # Another hidden layer
    Dense(10, activation='softmax')    # Output layer (10 classes for digits 0–9)
])

# 4️⃣ Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5️⃣ Train Model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 6️⃣ Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# 7️⃣ Make Predictions
predictions = model.predict(x_test)

# 8️⃣ Display some predictions
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title(f"Pred: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.show()
