from PIL import Image
import numpy as np
from random import randint

def insert_image(back, front, front_size, front_coordinate):
    background = Image.open(back).convert('RGB')
    background = background.resize((700, 700))
    background = np.array(background)
    front_img = Image.open(front).convert('RGB')
    front_img = front_img.resize(front_size)
    front_img = np.array(front_img)
    for h, line in enumerate(front_img):
        for w, pixel in enumerate(line):
            if ((int(pixel[0]) + int(pixel[2])) / 2) - int(pixel[1]) > 50:
                continue
            background[front_coordinate[1] + h][front_coordinate[0] + w] = pixel
    background = Image.fromarray(background)
    return background


def random_point(front_size):
    x_range = int(700-front_size[0]-1)
    y_range = int(700-front_size[1]-1)
    x = randint(0, x_range)
    y = randint(0, y_range)
    return (x, y)

def random_size(minimum, maximum):
    min_x, min_y = minimum
    max_x, max_y = maximum
    x = randint(min_x, max_x)
    y = randint(min_y, max_y)
    return (x, y)