
# coding: utf-8

# In[1]:


from utils.imports import *
from keras.layers.merge import add
from keras.utils.multi_gpu_utils import multi_gpu_model



# In[2]:


src = PATH['model_train']
src_val = PATH['model_val']
model_paths = PATH['model_paths']


# In[3]:


def identity_block(x,nb_filter,kernel_size=3):
    k1,k2,k3 = tuple(map(int,nb_filter))
    out = Convolution2D(k1,(1,1))(x)
    out = BatchNormalization(axis=-1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2,(kernel_size,kernel_size),border_mode='same')(out)
    out = BatchNormalization(axis=-1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3,(1,1))(out)
    out = BatchNormalization(axis=-1)(out)

    out = add([out,x])
    #out = merge([out,x],mode='sum')
    out = Activation('relu')(out)
    return out

def conv_block(x, nb_filter, kernel_size=3, strides=(2, 2)):
    k1, k2, k3 = tuple(map(int, nb_filter))


    out = Convolution2D(k1, (1, 1), strides=strides)(x)
    out = BatchNormalization(axis=-1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2,(kernel_size,kernel_size),border_mode='same')(out)
    out = BatchNormalization(axis=-1)(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3, (1, 1))(out)
    out = BatchNormalization(axis=-1)(out)

    shortcut = Convolution2D(k3, (1, 1), strides=strides)(x)
    shortcut = BatchNormalization(axis=-1)(shortcut)
    out = add([out, shortcut])
    #out = merge([out, shortcut],mode='sum')
    out = Activation('relu')(out)
    return out

def unet_model(dropout_rate,learn_rate,width):
    inputs = Input((1, 512,512))       
    # Normalization
    #x = Lambda(lambda x: x / 255, name='pre-process')(inputs)
    x = Convolution2D(width, (3, 3), strides=(1, 1), padding='same',activation="elu")(inputs)
    x = BatchNormalization(axis=-1)(x)
    # Block 1
    c1 = conv_block(x,(width/2, width/2,width),strides=(1,1))
    c1 = identity_block(c1,(width/2, width/2,width))      # 128,512,512
   
    #p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    
    # Block 2
    c2 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(c1)
    c2 = conv_block(c2,(width,width,width*2),strides=(2,2))
    c2 = identity_block(c2,(width,width,width*2))    # 256,256,256
    #p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    
    # Block 3
    c3 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(c2)
    c3 = conv_block(c3,(width*2, width*2,width*4),strides=[2,2])
    c3 = identity_block(c3,(width*2, width*2,width*4))   # 80, 120, 3
    #p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    print(c3)
    # Block 4
    c4 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(c3)
    c4 = conv_block(c4,(width*4, width*4,width*8),strides=[2,2])
    c4 = identity_block(c4,(width*4, width*4,width*8))   # 40, 60, 3
    #p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    # Block 5
    c5 = Convolution2D(width*16, (3, 3), padding="same", activation="elu")(c4)
    c5 = conv_block(c5,(width*8, width*8,width*16),strides=[2,2])
    c5 = identity_block(c5,(width*8, width*8,width*16)) # 20, 30, 3
    
    # Block 6
    u6 = concatenate([UpSampling2D(size=(2, 2))(c5), c4], axis=1)
    c6 = SpatialDropout2D(dropout_rate)(u6)
    c6 = Convolution2D(width*8, (3, 3), padding="same", activation="elu")(c6)
    c6 = conv_block(c6,(width*4, width*4,width*8),strides=[1,1])
    c6 = identity_block(c6,(width*4, width*4,width*8))

    # Block 7
    u7 = concatenate([UpSampling2D(size=(2, 2))(c6), c3], axis=1)
    # u7 = merge([UpSampling2D(size=(2, 2))(c6), c3], mode='concat', concat_axis=1)
    c7 = SpatialDropout2D(dropout_rate)(u7)
    c7 = Convolution2D(width*4, (3, 3), padding="same", activation="elu")(c7)
    c7 = conv_block(c7,(width*2, width*2,width*4),strides=[1,1])
    c7 = identity_block(c7,(width*2, width*2,width*4))

    # Block 8
    u8 = concatenate([UpSampling2D(size=(2, 2))(c7), c2], axis=1)
    # u8 = merge([UpSampling2D(size=(2, 2))(c7), c2], mode='concat', concat_axis=1)
    c8 = SpatialDropout2D(dropout_rate)(u8)
    c8 = Convolution2D(width*2, (3, 3), padding="same", activation="elu")(c8)
    c8 = conv_block(c8,(width, width,width*2),strides=[1,1])
    c8 = identity_block(c8, (width, width,width*2))

    # Block 9
    u9 = concatenate([UpSampling2D(size=(2, 2))(c8), c1], axis=1)
    c9 = SpatialDropout2D(dropout_rate)(u9)
    c9 = Convolution2D(width, (3, 3), padding="same", activation="elu")(c9)
    c9 = conv_block(c9,(width/2, width/2,width),strides=[1,1])
    c9 = identity_block(c9,(width/2, width/2,width))
    c10 = Convolution2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(input=inputs, output=c10)
    #model.summary()
    model.compile(optimizer=Adam(lr=learn_rate), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=SGD(lr=learn_rate, momentum=0.9, nesterov=True), loss=dice_coef_loss, metrics=[dice_coef])
    
    #plot_model(model, to_file='model1.png',show_shapes=True)

    return model

def unet_fit(name, check_name = None):
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
        target_size=(512,512),
        color_mode="grayscale",
        batch_size=1)

    mask_generator = mask_datagen.flow_from_directory(
        src,
        class_mode=None,
        classes=['nodule'],
        seed=seed,
        target_size=(512,512),
        color_mode="grayscale",
        batch_size=1) 
    datagen_val = ImageDataGenerator()
    image_generator_val = datagen_val.flow_from_directory(
        src_val,
        class_mode=None,
        classes=['lung'],
        seed=seed,
        target_size=(512,512),
        color_mode="grayscale",
        batch_size=1)

    mask_generator_val = datagen_val.flow_from_directory(
        src_val,
        class_mode=None,
        classes=['nodule'],
        seed=seed,
        target_size=(512,512),
        color_mode="grayscale",
        batch_size=1) 
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator) 
    val_generator = zip(image_generator_val, mask_generator_val)
    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience = 20, 
                                   verbose = 1),
    ModelCheckpoint(model_paths + '{}.h5'.format(name), 
                        monitor='val_loss', 
                        verbose = 0, save_best_only = True)]
    
    if check_name is not None:
        check_model = model_paths + '{}.h5'.format(check_name)
        model = load_model(check_model, 
                           custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else:
        model = unet_model(dropout_rate = 0.30, learn_rate = 1e-5,width=64)

    model = multi_gpu_model(model,gpus=4)
    model.fit_generator(
        train_generator,
        epochs=300,
        verbose =1, 
        callbacks = callbacks,
        steps_per_epoch=1280,
        validation_data = val_generator,
        nb_val_samples = 256)
    return


# In[4]:


unet_fit('final_fenge_Resnet')


# In[ ]:




