import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import *
import PIL
from PIL import Image, ImageDraw
from matplotlib.widgets import Button
from matplotlib.backends.backend_agg import FigureCanvasAgg

from IPython.display import display, clear_output
from ipywidgets import widgets

# loading the data, normalizing it and converting the labels to one hot encoding
def load_data(data_set, n):
    (X_train, Y_train), (X_test, Y_test) = data_set
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_copy_plot = Y_train
    Y_train = tf.keras.utils.to_categorical(Y_train, 10)
    Y_test = tf.keras.utils.to_categorical(Y_test, 10)
    print("X_train shape", X_train.shape)
    print("y_train shape", Y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", Y_test.shape)
    
    # plotting 9 random images from the training set to visualize the data
    for i in range(n*n):
        plt.subplot(n,n,i+1)
        num = random.randint(0, len(X_train))
        plt.imshow(X_train[num], cmap='gray', interpolation='none')
        plt.title("Class {}".format(Y_copy_plot[num]))
        plt.tight_layout()
    plt.show()
    return X_train, Y_train, X_test, Y_test

# Create a 'CNN' or 'Perceptron' model and train it on the given dataset
def model_training(model, x_train, y_train, x_test, y_test, batch_size, epochs, model_type):
    
    if model_type == 'CNN':
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
    elif model_type == 'Perceptron':
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
    else :
        print("Invalid model type")
        return
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# User drawn digit using the tkinter library
def draw_mnist_digit():
    # Create a new Tkinter window
    root = Tk()

    # Create a new canvas with a size of 280x280 pixels
    canvas = Canvas(root, width=280, height=280, bg='white')  # Scaled 10x for better drawing
    canvas.pack()

    # Create a new image with a size of 280x280 pixels
    image = Image.new("RGB", (280, 280), "white")  # Scaled 10x for better drawing
    draw = ImageDraw.Draw(image)
    
    line_thickness = 20

    # Function to start drawing
    def start_draw(event):
        canvas.old_coords = event.x, event.y

    # Function to draw
    def draw_funct(event):
        x, y = event.x, event.y
        old_x, old_y = canvas.old_coords
        canvas.create_line(old_x, old_y, x, y, fill="black", width=line_thickness)  # Scaled 10x for better drawing
        draw.line([old_x, old_y, x, y], fill="black", width=line_thickness)  # Scaled 10x for better drawing
        canvas.old_coords = x, y

    # Function to end drawing
    def end_draw(event):
        root.destroy()  # This will destroy the Tkinter window
        
    # Bind the mouse events to the drawing functions
    canvas.bind('<Button-1>', start_draw)
    canvas.bind('<B1-Motion>', draw_funct)
    canvas.bind('<Return>', end_draw)  # Bind the 'Enter' key to the end_draw function

    # Start the Tkinter event loop
    root.mainloop()
    
    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    # Resize the image to 28x28 pixels
    resized_image = grayscale_image.resize((28, 28), Image.LANCZOS)
    # Convert the image to a numpy array
    mnist_compatible_data = np.array(resized_image)
    # Invert the colors (because MNIST has white digits on a black background)
    mnist_compatible_data = 255 - mnist_compatible_data
    # Normalize the data (because MNIST data is normalized)
    mnist_compatible_data = mnist_compatible_data / 255.0
    # Reshape the data to match the input shape of the model
    mnist_compatible_data = mnist_compatible_data.reshape(1, 28, 28, 1)
    # Return the MNIST-compatible image
    return mnist_compatible_data


# Predict the digit drawn by the user using the previously trained model
def predict_digit(image, model):
    
    # Make the prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    # Print the prediction to check its values
    print(prediction)

    return predicted_digit

'''

# in this example we are loading the mnist dataset from keras    
given_data = tf.keras.datasets.mnist.load_data()

# number of random images to plot from the training set (example_size^2 images are plotted)
example_size = 2

# creating a sequential model
model = tf.keras.models.Sequential()

# number of epochs, batch size and type of network to train ('CNN' or 'Perceptron')
num_epochs = 25
actual_batch_size = 64
network_type = 'CNN'

# loading the given data and plotting random images from the training set
(x_train, y_train, x_test, y_test) = load_data(given_data, example_size)


# training the model on the given data
model_training(model, x_train, y_train, x_test, y_test, actual_batch_size, num_epochs, network_type)

# saving the model for future use
model.save('mnist-project-model.h5')
'''

# loading the previously saved model
loaded_model = tf.keras.models.load_model('mnist-project-model.h5')
print("Model loaded successfully\n")

# printing the model summary
print("Model Summary\n")
loaded_model.summary()

'''
# testing the digit drawing function
image = draw_mnist_digit()
'''

# predicting the digit drawn by the user
drawn_digit = draw_mnist_digit()
plt.imshow(drawn_digit.reshape(28, 28), cmap='gray')
plt.show()
prediction = predict_digit(drawn_digit, loaded_model)
print("\n\n\n=====================================\n\n\n")
print("The predicted digit is: ", prediction)
print("\n\n\n=====================================\n\n\n")