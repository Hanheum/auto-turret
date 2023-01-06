import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from time import time

#=======================================================================================================================
primary_interpreter = tf.lite.Interpreter(model_path='C:\\Users\\chh36\\Desktop\\turret\\turret_model1.tflite')
primary_interpreter.allocate_tensors()
primary_input_details = primary_interpreter.get_input_details()
primary_output_details = primary_interpreter.get_output_details()

secondary_interpreter = tf.lite.Interpreter(model_path='C:\\Users\\chh36\\Desktop\\turret\\turret_model2.tflite')
secondary_interpreter.allocate_tensors()
secondary_input_details = secondary_interpreter.get_input_details()
secondary_output_details = secondary_interpreter.get_output_details()
#=======================================================================================================================

#=======================================================================================================================
def scan(image, size):
    width, height = image.size
    steps_w = round(width/size)
    steps_w = steps_w*2 - 1
    steps_h = round(height/size)
    steps_h = steps_h*2 - 1
    moving_size = round(size/2)
    coordinates = []
    for w_step in range(int(steps_w)):
        x_start = moving_size*w_step
        x_end = x_start+size
        for h_step in range(int(steps_h)):
            y_start = moving_size*h_step
            y_end = y_start+size
            img = image.crop((x_start, y_start, x_end, y_end))
            img = img.resize((100, 100))
            img = np.array(img, dtype=np.float32)
            img = np.reshape(img, [1, 100, 100, 3])
            primary_interpreter.set_tensor(primary_input_details[0]['index'], img)
            primary_interpreter.invoke()
            prediction = primary_interpreter.get_tensor(primary_output_details[0]['index'])[0]
            if np.argmax(prediction) == 0:
                coordinate = (x_start, y_start)
                coordinates.append(coordinate)
    print('{} points detected'.format(len(coordinates)))
    return coordinates

def coordinate_trace(image, start_point, size):
    img = image.crop((*start_point, start_point[0]+size, start_point[1]+size))
    img = img.convert('L')
    img = img.resize((100, 100))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, [1, 100, 100, 1])
    secondary_interpreter.set_tensor(secondary_input_details[0]['index'], img)
    secondary_interpreter.invoke()
    prediction = secondary_interpreter.get_tensor(secondary_output_details[0]['index'])[0]*(size/100)
    coordinate = (start_point[0]+prediction[0], start_point[1]+prediction[1])
    return coordinate

def analyze(image_dir, scan_size, image_size, graph=True):
    image = Image.open(image_dir).convert('RGB')
    image = image.resize(image_size)
    if graph:
        plt.imshow(image)

    start_time = time()
    scaned_coordinates = scan(image, scan_size)

    coordinates = []
    for scaned_coordinate in scaned_coordinates:
        coordinate = coordinate_trace(image, scaned_coordinate, scan_size)
        coordinates.append(coordinate)
        if graph:
            plt.plot([coordinate[0]], [coordinate[1]], 'ro')

    end_time = time()
    duration = end_time - start_time
    print('duration: {} seconds'.format(duration))

    if graph:
        plt.show()

    return coordinates

def distance2D(A, B):
    Ax, Ay = A
    Bx, By = B
    distance = ((Bx-Ax)**2+(By-Ay)**2)**0.5
    return distance

def radius_check(r, coordinates):
    returning_coordinates = []
    for a, comparing in enumerate(coordinates):
        count = 0
        for b, target in enumerate(coordinates):
            dis = distance2D(comparing, target)
            if dis <= r:
                count += 1
        if count >= 3:
            returning_coordinates.append(comparing)
    return returning_coordinates
#=======================================================================================================================

image_dir = 'C:\\Users\\chh36\\Desktop\\test_img2.jpg'
scan_size = 500
image_size = (2000, 1000)

img = Image.open(image_dir)
img = img.resize(image_size)
plt.imshow(img)

final_coordinates = []

for i in range(11):
    scan_size += 10*i
    coordinates = analyze(image_dir, scan_size, image_size, graph=False)
    for a in coordinates:
        final_coordinates.append(a)

radius = 50
repeat = True
while repeat:
    try:
        true_coordinates = radius_check(radius, final_coordinates)
        size = len(true_coordinates)
        one_coordinate = sum(np.array(true_coordinates)) / size
        plt.plot([one_coordinate[0]], [one_coordinate[1]], 'ro')
        plt.show()
        repeat = False
    except:
        radius += 10