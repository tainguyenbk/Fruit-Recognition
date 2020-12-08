import utils
import numpy as np
from cv2 import cv2
import random
import os

image_path = "dataset/train"
new_image_path = "dataset/new_train"

image_labels = os.listdir(image_path)
image_labels.sort()
print(image_labels)

for name in image_labels:
    dir = os.path.join(image_path, name)
    new_dir = os.path.join(new_image_path, name)
    for x in range(1, 11):
        file = dir + "/" + "image (" + str(x) + ").jpg"
        count = 1
        image = cv2.imread(file)
        for i in range(1, 8 + 1):#8
            new_image = utils.foo(utils.rotate_image(image, random.uniform(0, 360)))
            new_file = new_dir + "/" + str(count + 40 * (x - 1)) + ".jpg"
            cv2.imwrite(new_file, new_image)
            print(count + 40 * (x - 1))
            count += 1
        for i in range(1, 7 + 1):#7
            new_image = utils.contras_image(image, random.uniform(80, 200) / 100, random.uniform(0, 50) / 10)
            new_file = new_dir + "/" + str(count + 40 * (x - 1)) + ".jpg"
            cv2.imwrite(new_file, new_image)
            print(count + 40 * (x - 1))
            count += 1
        for i in range(1, 7 + 1):#7
            new_image = utils.random_shadow(image)
            new_file = new_dir + "/" + str(count + 40 * (x - 1)) + ".jpg"
            cv2.imwrite(new_file, new_image)
            print(count + 40 * (x - 1))
            count += 1
        for i in range(-1, 2):#3
            new_image = utils.flip_image(image, i)
            new_file = new_dir + "/" + str(count + 40 * (x - 1)) + ".jpg"
            cv2.imwrite(new_file, new_image)
            print(count + 40 * (x - 1))
            count += 1
        for i in range(1, 8 + 1):#8
            _image = utils.foo(utils.rotate_image(image, random.uniform(0, 360)))
            new_image = utils.contras_image(_image, random.uniform(80, 200) / 100, random.uniform(0, 50) / 10)
            new_file = new_dir + "/" + str(count + 40 * (x - 1)) + ".jpg"
            cv2.imwrite(new_file, new_image)
            print(count + 40 * (x - 1))
            count += 1
        for i in range(1, 7 + 1): #7
            _image = utils.foo(utils.rotate_image(image, random.uniform(0, 360)))
            new_image = utils.random_shadow(_image)
            new_file = new_dir + "/" + str(count + 40 * (x - 1)) + ".jpg"
            cv2.imwrite(new_file, new_image)
            print(count + 40 * (x - 1))
            count += 1
