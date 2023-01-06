import tensorflow as tf
import numpy as np
from PIL import Image

cor_dir = input('coordinates directory:')
dataset_dir = input('dataset images directory:')
title_coordinate = open(cor_dir, 'r').readlines()

titles = []
val_titles = []
coordinates = []
val_coordinates = []
train_images = []
val_images = []

val_split = 0.2

for i, set in enumerate(title_coordinate):
    no_enter = set.split('\n')[0]
    three = no_enter.split('|')
    title = three[0]
    coordinate = [float(three[1])/7, float(three[2])/7]

    if (i/len(title_coordinate))<val_split:
        val_titles.append(title)
        val_coordinates.append(coordinate)
    else:
        titles.append(title)
        coordinates.append(coordinate)

coordinates = np.asarray(coordinates)
val_coordinates = np.asarray(val_coordinates)

for title in titles:
    image = Image.open(dataset_dir+'\\'+title+'.png').convert('L')
    image = image.resize((100, 100))
    image = np.array(image)
    train_images.append(image)

for title in val_titles:
    image = Image.open(dataset_dir+'\\'+title+'.png').convert('L')
    image = image.resize((100, 100))
    image = np.array(image)
    val_images.append(image)

train_images = np.asarray(train_images)
val_images = np.asarray(val_images)

network = [
    tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), input_shape=(100, 100, 1), activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=4, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.Conv2D(filters=4, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='relu')
]

model = tf.keras.Sequential(network)
model.compile(optimizer='adam', loss='mse')
model.fit(x=train_images, y=coordinates, validation_data=(val_images, val_coordinates), epochs=50)
model.save('C:\\Users\\chh36\\Desktop\\turret_model2')

