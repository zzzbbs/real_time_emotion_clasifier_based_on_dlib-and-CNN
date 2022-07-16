import dlib
from keras.models import load_model
import numpy as np
import cv2
from utils import load_image, detect_faces, get_coordinates, preprocess_input, draw_bounding_box,draw_text

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# 初始化dilc的人脸预测器（HOG）然后 创建面部标志预测器
#image_path = r'..\PrivateTest\anger\32303.jpg'
image_path = 'liudehua.jpg'
detector = dlib.get_frontal_face_detector()
emotion_model_path = '../simpler_CNN2.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'sad',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}

emotion_classifier = load_model(emotion_model_path, compile=False)
# 获取模型输入图像的宽和高尺寸
emotion_target_size = emotion_classifier.input_shape[1:3]

# 加载原始图像
rgb_image = load_image(image_path,grayscale=False)
gray_image = load_image(image_path,grayscale=True)


image = cv2.imread(image_path)
# image = cv2.imread('.\images\liudehua.jpg')
'''
(h, w) = image.shape[:2]
width = 500
r = width / float(w)
dim = (width, int(h * r))
# 加载输入图像，调整大小，并将其转换为灰度图
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

'''

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)
print(rects)

for (i, rect) in enumerate(rects):
    x1,y1,width,height = rect_to_bb(rect)
    x2 = x1+width
    y2 = y1+height
    print(x1,y1,x2,y2)
    #x1, x2, y1, y2 = get_coordinates(rect_to_bb(rect))
    # 抠出 人脸  数组
    gray_face = gray_image[y1:y2, x1:x2]
    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))

    except:
        print("转换失败")
        continue

    # 归一化
    gray_face = preprocess_input(gray_face)

    gray_face = np.expand_dims(gray_face, 0)

    # (1, 48, 48, 1)  # (图片数量, 高, 宽, 通道数)

    gray_face = np.expand_dims(gray_face, -1)

    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))

    emotion_text = emotion_labels[emotion_label_arg]

    print('emotion_text = ', emotion_text)

    # 画边框
    color = (0, 0, 255)  # opencv: bgr格式定义颜色
    draw_bounding_box(rect_to_bb(rect), image, color)
    draw_text(rect_to_bb(rect), image, emotion_text, color, 0, rect_to_bb(rect)[3] + 30, 1, 2)
    # 在灰度图像中进行人脸检测
   # color = (0,0,255)
    #draw_bounding_box(rect_to_bb(rect), image, color)
cv2.imshow('img',image)
cv2.waitKey(0)
