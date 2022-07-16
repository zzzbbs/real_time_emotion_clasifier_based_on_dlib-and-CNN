import cv2
from keras.models import load_model
from utils import load_image, detect_faces, get_coordinates, preprocess_input, draw_bounding_box,draw_text
import numpy as np
image_path = 'test.png'
detection_model_path = 'haarcascade_frontalface_alt.xml'
emotion_model_path = '../simpler_CNN2.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'sad',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
# 加载人脸模型
face_decttion = cv2.CascadeClassifier(detection_model_path)

emotion_classifier = load_model(emotion_model_path, compile=False)
# 获取模型输入图像的宽和高尺寸
emotion_target_size = emotion_classifier.input_shape[1:3]

# 加载原始图像
rgb_image = load_image(image_path,grayscale=False)
gray_image = load_image(image_path,grayscale=True)
# 使用keras

# cv2加载原始图像
oringinalImg = cv2.imread(image_path)
cv2.imshow('cv2',oringinalImg)

# 去掉维度为1的维度, (只留下宽和高, 去掉灰度维度)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

# 检测到了所有的人脸
faces = detect_faces(face_decttion, gray_image)

# 处理每一个脸
for face_coordinates in faces:
    x1, x2, y1, y2  = get_coordinates(face_coordinates)
    # 抠出 人脸  数组
    gray_face = gray_image[y1:y2, x1:x2]
    cv2.imshow('face',gray_face)
    cv2.waitKey(0)
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
    color = (0, 0, 255) # opencv: bgr格式定义颜色
    draw_bounding_box(face_coordinates, oringinalImg, color)
    draw_text(face_coordinates, oringinalImg,emotion_text, color, 0, face_coordinates[3] + 30, 1, 2)
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imshow('image', oringinalImg)
cv2.waitKey(0)
cv2.imwrite('images/predict4.jpg', bgr_image)
