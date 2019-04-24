
# coding: utf-8

# In[1]:

import sys
sys.path.append('..')
# from utils.imports import *
import tensorflow as tf
import pandas as pd
from utils.paths import PATH
import os
import numpy as np
from tqdm import tqdm
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)
from python.model_Inception import get_Inception_classifier

# In[2]:


csv_path = PATH['annotations_test']
src = PATH['model_test_pred']
pred_csv_path = PATH['model_test_pred']
data_path = PATH['src_test']


# In[3]:


model_paths = PATH['model_paths']
# model_final = PATH['model_final']
model_final = '/devdata1/ding/data/TianChi/ali/Process/model/'

# In[4]:
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)


def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

from python.model_VGG import get_net
from keras import backend as K
K.set_image_dim_ordering('th')
model_cube_30,model_s = get_net()
model_cube_30.load_weights(model_paths + 'Fenge_32_32_32_0704.h5')
model_cube_30 = model_s
# model_20 = load_model(model_final + 'Fenge_32_32_32_inception.h5')
# model_30 = load_model(model_final + 'Fenge_32_32_32_densenet.h5')
# model_40 = load_model(model_final + 'Fenge_26_40_40.h5')


# In[5]:


test_pred_0 = pd.read_csv(pred_csv_path + "1final_test_result.csv")


# In[6]:


patients = [x for x in os.listdir(pred_csv_path) if 'orig' in x]    


# In[ ]:


test_pred_0["file"] = test_pred_0["seriesuid"].map(lambda file_name: get_filename(patients, file_name))
test_pred_0 = test_pred_0.dropna()


# In[ ]:


probability_30_30_30_cube = []
# probability_06_20_20 = []
# probability_10_30_30 = []
# probability_26_40_40 = []
average = []

for img_file in tqdm(sorted(patients)):
    mini_df_anno = test_pred_0[test_pred_0["file"]==img_file] #get all nodules associate with file
    if mini_df_anno.shape[0]>0: # some files may not have a nodule--skipping those 
        # load the data once        
        patient_id = img_file[:-9]
        img_array = np.load(src + img_file)
        pos_annos = pd.read_csv(src + img_file[:-9] + '_annos_pos.csv')
        origin = np.array([pos_annos.loc[0]['origin_x'],pos_annos.loc[0]['origin_y'],pos_annos.loc[0]['origin_z']]) 
        spacing = np.array([pos_annos.loc[0]['spacing_x'],pos_annos.loc[0]['spacing_y'],pos_annos.loc[0]['spacing_z']])
        img_array = normalize(img_array)                
        for node_idx1, cur_row1 in mini_df_anno.iterrows():       
            node_x = cur_row1["coordX"]
            node_y = cur_row1["coordY"]
            node_z = cur_row1["coordZ"]
            diam = cur_row1["diameter_mm"]
            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint(np.absolute(center-origin)/spacing)            
            new_x = int(v_center[0])
            new_y = int(v_center[1])
            new_z = int(v_center[2])
            
            if new_z<18 or new_x<18 or new_y<18 or new_x+18>img_array.shape[2] or new_y+18>img_array.shape[1] or new_z+18>img_array.shape[0]:
                cls_result_cube_30 = int(0)
            else:
   
                trainX_cube_30 =  img_array[new_z - 16: new_z + 16,
                                    new_y - 16 : new_y + 16,
                                    new_x - 16 : new_x + 16]
            
                trainX_cube_30=np.expand_dims(trainX_cube_30,0)
                trainX_cube_30=np.expand_dims(trainX_cube_30,0)
            
                cls_result_cube_30 = model_cube_30.predict(trainX_cube_30)[0][1]
            probability_30_30_30_cube.append(cls_result_cube_30)

            
            # if new_z<3 or new_x<10 or new_y<10 or new_x+10>img_array.shape[2] or new_y+10>img_array.shape[1] or new_z+3>img_array.shape[0]:
            #     cls_result20 = int(0)
            # else:
            #
            #     trainX_20 =  img_array[new_z - 3: new_z + 3,
            #                         new_y - 10 : new_y + 10,
            #                         new_x - 10 : new_x + 10]
            #
            #     trainX_20=np.expand_dims(trainX_20,0)
            #     trainX_20=np.expand_dims(trainX_20,0)
            #
            #     cls_result_20 = model_20.predict(trainX_20)[0][1]
            # probability_06_20_20.append(cls_result_20)
            
            # if new_z<5 or new_x<15 or new_y<15 or new_x+15>img_array.shape[2] or new_y+15>img_array.shape[1] or new_z+5>img_array.shape[0]:
            #     cls_result30 = int(0)
            # else:
            #
            #     trainX_30 =  img_array[new_z - 5: new_z + 5,
            #                         new_y - 15 : new_y + 15,
            #                         new_x - 15 : new_x + 15]
            #
            #     trainX_30=np.expand_dims(trainX_30,0)
            #     trainX_30=np.expand_dims(trainX_30,0)
            #
            #     cls_result_30 = model_30.predict(trainX_30)[0][1]
            # probability_10_30_30.append(cls_result_30)
            
            
            # if new_z<13 or new_x<20 or new_y<20 or new_x+20>img_array.shape[2] or new_y+20>img_array.shape[1] or new_z+13>img_array.shape[0]:
            #     cls_result40 = int(0)
            # else:
            #
            #     trainX_40 =  img_array[new_z - 13: new_z + 13,
            #                         new_y - 20 : new_y + 20,
            #                         new_x - 20 : new_x + 20]
            #
            #     trainX_40=np.expand_dims(trainX_40,0)
            #     trainX_40=np.expand_dims(trainX_40,0)
            #
            #     cls_result_40 = model_40.predict(trainX_40)[0][1]
            # probability_26_40_40.append(cls_result_40)
            
            avg = cls_result_cube_30
            average.append(avg)


# In[ ]:


probability_30_30_30_cube = np.array(probability_30_30_30_cube)
probability_30_30_30_cube = probability_30_30_30_cube.clip(0.005,0.995)
probability_30_30_30_cube = probability_30_30_30_cube.round(3)
test_pred_0['probability_30_30_30_cube'] = probability_30_30_30_cube

#
# probability_06_20_20 = np.array(probability_06_20_20)
# probability_06_20_20 = probability_06_20_20.clip(0.005,0.995)
# probability_06_20_20 = probability_06_20_20.round(3)
# test_pred_0['probability_06_20_20'] = probability_06_20_20
#
# probability_10_30_30 = np.array(probability_10_30_30)
# probability_10_30_30 = probability_10_30_30.clip(0.005,0.995)
# probability_10_30_30 = probability_10_30_30.round(3)
# test_pred_0['probability_10_30_30'] = probability_10_30_30
#
# probability_26_40_40 = np.array(probability_26_40_40)
# probability_26_40_40 = probability_26_40_40.clip(0.005,0.995)
# probability_26_40_40 = probability_26_40_40.round(3)
# test_pred_0['probability_26_40_40'] = probability_26_40_40

average = np.array(average)
average = average.clip(0.005,0.995)
average = average.round(3)
test_pred_0['probability'] = average


# In[ ]:


test_pred_0.to_csv(csv_path + "0final.csv", index=False)

