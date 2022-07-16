from tensorflow import keras
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU

from keras.layers import Convolution2D, BatchNormalization, AveragePooling2D, Dropout, Flatten, Dense, Activation
def simple_CNN(input_shape, num_classes):
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, padding='same', input_shape=input_shape))# 16 个 7*7滤波器
    model.add(PReLU()) #激活函数
    model.add(BatchNormalization()) #激活值规范化
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2),padding='same'))# 采样因子(5,5) 步长(2,2)
    model.add(Dropout(.2)) # dropout将在训练过程中按一定概率随机断开输入神经元，用于防止过拟合



    model.add(Convolution2D(32, 4, 4, padding='same', input_shape=input_shape))# 32 个 5*5滤波器
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2), padding='same'))
    model.add(Dropout(.2))



    model.add(Convolution2D(64, 5, 5, padding='same', input_shape=input_shape))# 32 个 3*3 滤波器
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(.2))

    # 展平
    model.add(Flatten()) #多维输入一维化，从卷积层到全连接层的过渡
    model.add(Dense(2048))#输出维度1028
    model.add(PReLU())
    model.add(Dropout(.4))
    model.add(Dense(1024))
    model.add(PReLU())
    model.add(Dropout(.4))
    model.add(Dense(num_classes))# 输出维度num_classes
    model.add(Activation('softmax'))

    return model

if __name__ == '__main__':
    input_shape = (48, 48, 1)
    num_classes = 7
    model = simple_CNN(input_shape, num_classes)
    model.summary()
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
