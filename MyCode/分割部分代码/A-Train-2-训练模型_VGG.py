# coding: utf-8

# In[1]:

import sys
sys.path.append('..')
from utils.imports import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)

# ## 训练模型

# In[2]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

session = tf.Session(config=config)

src = PATH['model_train']
src_val = PATH['model_val']
model_paths = PATH['model_paths']


# In[ ]:


# In[3]:定义unet模型


def unet_model(dropout_rate, learn_rate, width):
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(width, (3, 3), padding="same", activation="elu")(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Convolution2D(width, (3, 3), padding="same", activation="elu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(width * 2, (3, 3), padding="same", activation="elu")(pool1)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Convolution2D(width * 2, (3, 3), padding="same", activation="elu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(width * 4, (3, 3), padding="same", activation="elu")(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Convolution2D(width * 4, (3, 3), padding="same", activation="elu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(width * 8, (3, 3), padding="same", activation="elu")(pool3)
    conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Convolution2D(width * 8, (3, 3), padding="same", activation="elu")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(width * 16, (3, 3), padding="same", activation="elu")(pool4)
    conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Convolution2D(width * 16, (3, 3), padding="same", activation="elu")(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = SpatialDropout2D(dropout_rate)(up6)
    conv6 = Convolution2D(width * 8, (3, 3), padding="same", activation="elu")(conv6)
    conv6 = Convolution2D(width * 8, (3, 3), padding="same", activation="elu")(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = SpatialDropout2D(dropout_rate)(up7)
    conv7 = Convolution2D(width * 4, (3, 3), padding="same", activation="elu")(conv7)
    conv7 = Convolution2D(width * 4, (3, 3), padding="same", activation="elu")(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = SpatialDropout2D(dropout_rate)(up8)
    conv8 = Convolution2D(width * 2, (3, 3), padding="same", activation="elu")(conv8)
    conv8 = Convolution2D(width * 2, (3, 3), padding="same", activation="elu")(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = SpatialDropout2D(dropout_rate)(up9)
    conv9 = Convolution2D(width, (3, 3), padding="same", activation="elu")(conv9)
    conv9 = Convolution2D(width, (3, 3), padding="same", activation="elu")(conv9)
    conv10 = Convolution2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(input=inputs, output=conv10)
    # model.summary()

    model.compile(optimizer=Adam(lr=learn_rate), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=SGD(lr=learn_rate, momentum=0.9, nesterov=True), loss=dice_coef_loss, metrics=[dice_coef])

    # plot_model(model, to_file='model1.png',show_shapes=True, show_layer_names=False)
    return model


def unet_fit(name, check_name=None):
    data_gen_args = dict(rotation_range=30.,     #数据增强
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,

                         )
    from keras.preprocessing.image import ImageDataGenerator
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_generator = image_datagen.flow_from_directory(  #送入训练集的肺部图像路径、尺寸
        src,
        class_mode=None,
        classes=['lung'],
        seed=seed,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=1)

    mask_generator = mask_datagen.flow_from_directory(#送入训练集的肺结节图像的路径、尺寸
        src,
        class_mode=None,
        classes=['nodule'],
        seed=seed,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=1)
    datagen_val = ImageDataGenerator()
    image_generator_val = datagen_val.flow_from_directory(#送入验证集的肺部图像的路径、尺寸
        src_val,
        class_mode=None,
        classes=['lung'],
        seed=seed,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=1)

    mask_generator_val = datagen_val.flow_from_directory(#送入验证集的肺部结节mask图像路径、尺寸
        src_val,
        class_mode=None,
        classes=['nodule'],
        seed=seed,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=1)
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)#训练集生成器
    val_generator = zip(image_generator_val, mask_generator_val)#验证集生成器
    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience=20,#回调函数，当val_loss值的上一次训练没有下降，经过20次后停止训练
                               verbose=1),
                 ModelCheckpoint(model_paths + '{}.h5'.format(name),#每个epoch后保存模型到h5文件中
                                 monitor='val_loss',
                                 verbose=0, save_best_only=True)]

    if check_name is not None: #加载模型
        check_model = model_paths + '{}.h5'.format(check_name)
        model = load_model(check_model,
                           custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else: #设置了dropout速率、学习速率以及通道宽度
        model = unet_model(dropout_rate=0.30, learn_rate=1e-5, width=128)
    model.fit_generator( #共设置了300个epoch，每个epoch有256个样本
        train_generator,
        epochs=300,
        verbose=1,
        callbacks=callbacks,
        steps_per_epoch=1280,
        validation_data=val_generator,
        nb_val_samples=256)
    return


# In[ ]:


# unet_fit('final_fenge_170626')
unet_fit('final_fenge_VGG', 'final_fenge_VGG')

# In[ ]:




