import cv2
# pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple/
import pandas as pd
import numpy as np
import os


emotions = {
    '0': 'anger',  # 生气
    '1': 'disgust',  # 厌恶
    '2': 'fear',  # 恐惧
    '3': 'happy',  # 开心
    '4': 'sad',  # 伤心
    '5': 'surprised',  # 惊讶
    '6': 'normal',  # 中性
}
#创建文件夹
def createDir(dir):
    if os.path.exists(dir) == False:
        os.makedirs(dir)
createDir("../imgs")
def saveImageFromFer2013(file):
    #读取csv文件
    faces_data = pd.read_csv(file)
    # print(faces_data.head(10))
    # print(faces_data.columns) #['emotion', 'pixels', 'Usage']
    # print(faces_data.shape)(35887, 3)
    # print(len(faces_data)) # 35887
    #遍历csv文件内容，并将图片数据按分类保存
    print(faces_data.head())
    for index in range(len(faces_data)):
        #解析每一行csv文件内容
        emotion_data = faces_data.iloc[index, 0]
        # print(emotion_data)
        image_data = faces_data.iloc[index, 1]
        # print(image_data)
        # print(type(image_data))
        usage_data = faces_data.iloc[index, 2]

        #  ['222', '222', '210'] 变成[222, 222, 210]
        data = list(map(float, image_data.split()))
        #将图片数据转换成48*48
        image = np.array(data).reshape(48, 48)
        print(image.shape)
        #选择分类，并创建文件名
        dirName = usage_data
        emotionName = emotions[str(emotion_data)]
        # print(emotionName)
        #图片要保存的文件夹
        image_path = os.path.join(dirName, emotionName)
        print(image_path)
        # 创建“用途文件夹”和“表情”文件夹
        createDir(image_path)
        #图片文件名
        image_Name = os.path.join(image_path, str(index) + '.jpg')
        cv2.imwrite(image_Name, image)  #使用cv2实现图片与numpy数组的相互转化


saveImageFromFer2013('./datasets/fer2013/fer2013.csv')
