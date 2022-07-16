import cv2
import numpy as np
import pandas as pd
from keras.preprocessing import image

def load_image(image_path, grayscale=False, target_size=None):
    color_mode = 'grayscale'
    if grayscale==False:
        color_mode = 'rgb'
    else:
        grayscale=False
    pill_image = image.load_img(image_path, grayscale, color_mode, target_size)
    return image.img_to_array(pill_image)

def detect_faces(detect_model, gray_image_array):
    return detect_model.detectMultiScale(gray_image_array, 1.3, 5)

def get_coordinates(face_coordinates):
    x, y, width, height = face_coordinates
    return (x, x +width , y, y+height)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, width, height = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + width, y + height), color, 2)

def draw_text(face_coordinates, image_array,  text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = face_coordinates[:2]

    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def preprocess_input(data):
    x = np.array(data, dtype=np.float32)
    x = x/255.0
    return x
