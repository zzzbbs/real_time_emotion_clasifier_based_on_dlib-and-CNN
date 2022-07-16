import os
from keras.callbacks import ModelCheckpoint, CSVLogger
from model import simple_CNN
from utils import load_data, preprocess_input


data_path = '../datasets/fer2013/fer2013.csv'
model_save_path = '../trained_models/simpler_CNN2.hdf5'

#加载人脸表情训练数据和对应表情标签
faces, emotions = load_data(data_path)

#人脸数据归一化，将像素值从0-255映射到0-1之间
faces = preprocess_input(faces)
#得到表情分类个数
num_classes = emotions.shape[1]

#(48, 48, 1)
image_size = faces.shape[1:]


batch_size = 128
num_epochs = 1000

model = simple_CNN(image_size, num_classes)

#断点续训
if os.path.exists(model_save_path):
    model.load_weights(model_save_path)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")

#编译模型，categorical_crossentropy多分类选用
model.compile(optimizer='adam', loss='categorical_crossentropy',
                                        metrics=['accuracy'])

#记录日志
csv_logger = CSVLogger('training.log')

#保存检查点
model_checkpoint = ModelCheckpoint(model_save_path,
                                    'val_accuracy', verbose=1,
                                    save_best_only=True)

model_callbacks = [model_checkpoint, csv_logger]

#训练模型
model.fit(faces,emotions,batch_size,num_epochs,verbose=1,
                                    callbacks=model_callbacks,
                                    validation_split=.1,
                                    shuffle=True)

