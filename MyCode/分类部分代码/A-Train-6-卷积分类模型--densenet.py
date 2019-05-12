
# coding: utf-8

# In[1]:

import sys
sys.path.append('..')
# from utils.imports import *
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.applications import densenet
from keras.optimizers import SGD,Adam
from utils.paths import PATH
import os
import numpy as np
from tqdm import tqdm
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
from python.model_DenseNet import get_DenseNet_classifier

# 设置session
KTF.set_session(session)


# In[2]:



# In[3]:


output_path = PATH['cls_train_cube_30']
output_true = PATH['vgg_cls_train_cube_30_true']
output_false = PATH['vgg_cls_train_cube_30_false']
model_paths = PATH['model_paths']
model_final = PATH['model_final']


# In[4]:


def get_dirfiles(dir):
    file_list = []
    subset_path = os.listdir(dir)
    for _ in range(len(subset_path)):
        if subset_path[_] != '.DS_Store':
            file_list.append(os.path.join(dir , subset_path[_]))
    return file_list

def train_generator(output_true,output_false):
    train_nb = 80000
    file_list_true = get_dirfiles(output_true)
    file_list_false = get_dirfiles(output_false)        
    file_list_true = np.random.choice(file_list_true, train_nb,replace=False)
    file_list_false = np.random.choice(file_list_false, train_nb,replace=False)
    nb_true = len(file_list_true) + len(file_list_false) 
    print(nb_true)

#     pd.read_csv(nb_true, skiprows=(i for i in range(160000) if i % 5 == 0))

    
    sample = np.zeros([nb_true,32, 32, 32])
    labels = np.zeros([nb_true,2])
    for i in tqdm(range(len(file_list_true))):       
        cc= np.load(file_list_true[i])
        sample[i] = cc
        labels[i] = [0.,1.]
    for j in tqdm(range(len(file_list_false))):
        bb= np.load(file_list_false[j])
        sample[j+len(file_list_true)] = bb 
        labels[j+len(file_list_true)] = [1.,0.]
    sample = np.array(sample)

    sample = np.expand_dims(sample, axis=4)
    idx = np.arange(0,len(sample),1)
    np.random.shuffle(idx)
    sample = sample[idx]
    labels = labels[idx]
    return sample, labels


def fenlei_fit(name, load_check = False,batch_size=2, epochs=20,check_name = None):
    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience = 10, verbose = 1),
                 ModelCheckpoint((model_paths + '{}.h5').format(name),
                                 monitor='val_loss',
                                 verbose = 0,
                                 save_best_only = True)]
    if load_check:
        check_model = (model_paths + '{}.h5').format(check_name)
        model = load_model(check_model)
    else:
        #model = classifier((1, 32, 32, 32),128,(3, 3, 3), (2, 2, 2))
        model,_ = get_DenseNet_classifier()
    x,y = train_generator(output_true, output_false)
    model.fit(x,y, batch_size=batch_size, epochs=epochs,
              validation_split=0.2,verbose=1, callbacks=callbacks, shuffle=True) 
    return model


# In[5]:


fenlei_fit('Fenge_32_32_32_densenet_s', load_check = False, batch_size=120, epochs=100, check_name = 'Fenge_32_32_32_0703')


# In[6]:


file_list_true = get_dirfiles(output_true)
file_list_false = get_dirfiles(output_false)


# In[7]:


#model_pred = classifier((1, 36, 36, 36), (3, 3, 3), (2, 2, 2))
model = get_DenseNet_classifier()
model_pred = load_model(os.path.join(model_paths,'Fenge_32_32_32_densenet.h5'))


# In[8]:


file_list_true = np.random.choice(file_list_true, 1000)
file_list_false = np.random.choice(file_list_false, 1000)


# In[9]:



pre_input = []
for i in file_list_false[0:200]:
    a=np.load(i)
    pre_input.append(a)

pre_input = np.array(pre_input)
pre_input = np.expand_dims(pre_input,axis=4)
cc = model_pred.predict(pre_input)
count = 0
for i in cc:
    if i[0] > 0.9:
        count += 1
print(count*1.0/len(cc))


# In[10]:





# In[11]:


pre_input = []
for i in file_list_true[0:200]:
    a=np.load(i)
    pre_input.append(a)

pre_input = np.array(pre_input)
pre_input = np.expand_dims(pre_input,axis=4)
cc = model_pred.predict(pre_input)

count = 0


# In[12]:


for i in cc:
    if i[1] > 0.9:
        count += 1
print(count*1.0/len(cc))


# In[13]:





# ### def solo(input_shape, load_weight_path=None, features=False, mal=False):
#     width = 64
#     inputs = Input(shape=(1, 32, 32, 32), name="input_1")
#     x = inputs
#     x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), border_mode="same")(x)
#     #x = BatchNormalization(axis = 1)(x)
#     x = Convolution3D(width, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1))(x)
#     x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1')(x)
# 
#     # 2nd layer group
#     #x = BatchNormalization(axis = 1)(x)
#     x = Convolution3D(width*2, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1))(x)
#     x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2')(x)
#     x = Dropout(p=0.3)(x)
# 
#     # 3rd layer group
#     #x = BatchNormalization(axis = 1)(x)
#     x = Convolution3D(width*4, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1))(x)
#     #x = BatchNormalization(axis = 1)(x)
#     x = Convolution3D(width*4, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1))(x)
#     x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3')(x)
#     x = Dropout(p=0.4)(x)
# 
#     # 4th layer group
#     #x = BatchNormalization(axis = 1)(x)
#     x = Convolution3D(width*8, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1))(x)
#     #x = BatchNormalization(axis = 1)(x)
#     x = Convolution3D(width*8, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1),)(x)
#     x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4')(x)
#     x = Dropout(p=0.5)(x)
#        
#     last64 = Convolution3D(64, 2, 2, 2, activation="relu", name="last_64")(x)
#     out_class = Convolution3D(1, 1, 1, 1, activation="softmax", name="out_class_last")(last64)
#     out_class = Flatten(name="out_class")(x)
#     
#     out_class = Dense(2)(out_class) 
#     #out_class = BatchNormalization(axis = 1)(out_class)
#     out_class = Activation('softmax')(out_class)
#     
#     model = Model(input=inputs, output=out_class)
#     model.compile(optimizer=SGD(lr=1e-3, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
#     #model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy']) 
#     plot_model(model, to_file='model1.png',show_shapes=True, show_layer_names=False)
#     return model
# 
# solo((1, 32, 32, 32))
# import keras
# keras.applications.inception_v3.InceptionV3(include_top=True,
#                                             weights='imagenet',
#                                             input_tensor=None,
#                                             input_shape=None,
#                                             pooling=None,
#                                             classes=1000)