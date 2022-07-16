import pandas as pd
import numpy as np
def load_data(data_file):
    faces_data = pd.read_csv(data_file)
    pixels = faces_data['pixels'].to_list()
    # 对数据进行 one_hot 编码
    df = pd.get_dummies(faces_data['emotion'])
    emotions = df.values
    w, h = 48, 48
    faces = []
    for pixel_seq in pixels:
        face = list(map(int, pixel_seq.split()))
        face = np.array(face).reshape(w, h)
        faces.append(face)
    faces = np.array(faces)
    # print(faces.shape) # (35887, 48, 48)
    # 增加一个维度
    faces = np.expand_dims(faces, -1)
    # print(faces.shape) # (35887, 48, 48)

    return faces, emotions
def preprocess_input(data):
    x = np.array(data, dtype=np.float32)
    x = x/255.0
    return x
