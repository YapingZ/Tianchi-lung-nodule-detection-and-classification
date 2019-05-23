# 导入必要的库
from utils.imports import *
# 定义路径
src = PATH['model_train']
src_val = PATH['model_val']
model_paths = PATH['model_paths']
# 定义yolo模型
def yolo_model(dropout_rate,learn_rate, width):
    inputs = Input(1, 512, 512)
    Conv2D(32, (3, 3), strides=(1, 1), input_shape=inputs, padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(32, (1, 1), strides=(1, 1), input_shape=inputs, padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),

    Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),
    MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),

    Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(1024, (3, 3), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    Conv2D(1024, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    BatchNormalization(axis=-1),

    return model


def yolo_fit(name, check_name=None):
    data_gen_args = dict(rotation_range=30.,
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

    image_generator = image_datagen.flow_from_directory(
        src,
        class_mode=None,
        classes=['lung'],
        seed=seed,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=1)

    mask_generator = mask_datagen.flow_from_directory(
        src,
        class_mode=None,
        classes=['nodule'],
        seed=seed,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=1)
    datagen_val = ImageDataGenerator()
    image_generator_val = datagen_val.flow_from_directory(
        src_val,
        class_mode=None,
        classes=['lung'],
        seed=seed,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=1)

    mask_generator_val = datagen_val.flow_from_directory(
        src_val,
        class_mode=None,
        classes=['nodule'],
        seed=seed,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=1)
    # combine generators into one which yields image and masks
    train_generator = itertools.izip(image_generator, mask_generator)
    val_generator = itertools.izip(image_generator_val, mask_generator_val)
    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience=20,
                               verbose=1),
                 ModelCheckpoint(model_paths + '{}.h5'.format(name),
                                 monitor='val_loss',
                                 verbose=0, save_best_only=True)]

    if check_name is not None:
        check_model = model_paths + '{}.h5'.format(check_name)
        model = load_model(check_model,
                           custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else:
        model = yolo_model(dropout_rate=0.30, learn_rate=1e-5, width=128)
    model.fit_generator(
        train_generator,
        epochs=300,
        verbose=1,
        callbacks=callbacks,
        steps_per_epoch=1280,
        validation_data=val_generator,
        nb_val_samples=256)
    return

# 显示训练果
yolo_fit('final_fenge_yolo')