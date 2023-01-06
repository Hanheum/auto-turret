import tensorflow as tf
import numpy as np
from PIL import Image
from os import listdir

dataset_dir = input('dataset directory:')+'\\'
saving_dir = input('saving directory:')
categories = listdir(dataset_dir)

train_images, train_labels, val_images, val_labels = [], [], [], []

val_split = 0.2

for i, category in enumerate(categories):
    label = np.zeros([len(categories)])
    label[i] = 1
    titles = listdir(dataset_dir+category)
    total = len(titles)
    for a, title in enumerate(titles):
        image = Image.open(dataset_dir+category+'\\'+title).convert('RGB')
        image = image.resize((100, 100))
        image = np.array(image)

        if (a/total)<val_split:
            val_images.append(image)
            val_labels.append(label)
        else:
            train_images.append(image)
            train_labels.append(label)

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
val_images = np.asarray(val_images)
val_labels = np.asarray(val_labels)

print('='*10+'load completed'+'='*10)

network = [
    tf.keras.layers.Conv2D(filters=10, kernel_size=(5, 5), input_shape=(100, 100, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Conv2D(filters=10, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(categories), activation='softmax')
]

model = tf.keras.Sequential(network)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_images, y=train_labels, validation_data=(val_images, val_labels), epochs=20)

model.save(saving_dir)

