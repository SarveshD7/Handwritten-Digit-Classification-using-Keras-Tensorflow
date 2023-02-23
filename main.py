# Importing necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# # Getting dataset
mnist = keras.datasets.mnist

# Getting training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalization
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28, 28)))
# # Flattens all the 784 pixels of the image from grid to a single line
# model.add(keras.layers.Dense(128, activation='relu'))
# # Dense layer is a layer in which each neuron is connected with every other between two layers
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))
# # This is our last layer ie, the output layer using softmax activation function
#
# # Compiling the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Training the model and saving it
# model.fit(x_train, y_train, epochs=3)
# model.save('mnist.model')

model = keras.models.load_model('mnist.model')

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)
image_number = 1
while os.path.isfile(f"data/digit{image_number}.png"):
    img = cv2.imread(f"data/digit{image_number}.png")[:, :, 0] # Since we are not interested in numbers
    # img = np.array(img)
    img = np.invert(np.array([img]))
    # img = img.reshape(-1)
    # print(img.shape)
    prediction = model.predict(img)
    plt.title(f"This digit is probably a {np.argmax(prediction)}")
    plt.imshow(img[0])
    plt.show()
    image_number+=1
