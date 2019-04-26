import sys
sys.path.append('..')
from utils.imports import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.utils.multi_gpu_utils import multi_gpu_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)

def get_net(input_shape=(1, 32, 32, 32), load_weight_path=None, features=False, mal=False):
    width = 64
    inputs = Input(shape=(1, 32, 32, 32), name="input_1")
    x = inputs
    x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), border_mode="same")(x)
    # x = BatchNormalization(axis = 1)(x)
    x = Convolution3D(width, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1')(x)

    # 2nd layer group
    # x = BatchNormalization(axis = 1)(x)
    x = Convolution3D(width * 2, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2')(x)
    x = Dropout(p=0.3)(x)

    # 3rd layer group
    # x = BatchNormalization(axis = 1)(x)
    x = Convolution3D(width * 4, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1))(x)
    # x = BatchNormalization(axis = 1)(x)
    x = Convolution3D(width * 4, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3')(x)
    x = Dropout(p=0.4)(x)

    # 4th layer group
    # x = BatchNormalization(axis = 1)(x)
    x = Convolution3D(width * 8, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1))(x)
    # x = BatchNormalization(axis = 1)(x)
    x = Convolution3D(width * 8, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1), )(
        x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4')(x)
    x = Dropout(p=0.5)(x)
    print(x)

    last64 = Convolution3D(64, 2, 2, 2, activation="relu", name="last_64")(x)
    out_class = Convolution3D(1, 1, 1, 1, activation="softmax", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(x)

    out_class = Dense(2)(out_class)
    # out_class = BatchNormalization(axis = 1)(out_class)
    out_class = Activation('softmax')(out_class)

    model_s = Model(input=inputs, output=out_class)
    model = multi_gpu_model(model_s, gpus=4)
    model.compile(optimizer=SGD(lr=1e-3, momentum=0.9, nesterov=True), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, model_s
