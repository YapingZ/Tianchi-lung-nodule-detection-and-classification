
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')
from utils.imports import *
import tensorflow as tf
from utils.utils import normalize,resize,get_filename
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


csv_path = PATH['annotations_val']
src = PATH['model_train_pred']
pred_csv_path = PATH['model_train_pred']
data_path = PATH['src_train']


# In[3]:


model_paths = PATH['model_paths']
model_final = PATH['model_final']


# In[4]:


# model_cube_30 = load_model(model_paths + 'Fenge_32_32_32_densenet.h5')
from python.model_VGG import get_net
from keras import backend as K
K.set_image_dim_ordering('th')
model_cube_30,model_s = get_net()
model_cube_30.load_weights(model_paths + 'Fenge_32_32_32_0704.h5')
model_cube_30 = model_s
# In[5]:

#加载分类结果文件
#test_pred_0 = pd.read_csv(pred_csv_path + "1_vgg_final_result.csv")
test_pred_0 = pd.read_csv('/devdata1/ding/data/TianChi/ali/all_1final/1final_test_result_v.csv')


# In[6]:


seriesuids = pd.read_csv(PATH['annotations_val'] + "seriesuids.csv")
test_pred_0["file"] = test_pred_0["seriesuid"].map(lambda file_name:  get_filename(seriesuids['seriesuid'].values, file_name))
test_pred_0 = test_pred_0.dropna()


# In[7]:


patients = [x for x in os.listdir(pred_csv_path) if 'orig' in x]   


# In[8]:


test_pred_0["file"] = test_pred_0["seriesuid"].map(lambda file_name: get_filename(patients, file_name))
test_pred_0 = test_pred_0.dropna()


# In[9]:


def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    if res.shape != (32, 32, 32):
        res = resize(res,[32,32,32])
    return res



probability_30_30_30_cube_ = []

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
            #diam = cur_row1["diameter_mm"]
            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint(np.absolute(center-origin)/spacing)            
            new_x = int(v_center[0])
            new_y = int(v_center[1])
            new_z = int(v_center[2])  

            trainX_cube_30 = get_cube_from_img(img_array, new_x, new_y, new_z, 32)      
            trainX_cube_30=np.expand_dims(trainX_cube_30,0)
            trainX_cube_30=np.expand_dims(trainX_cube_30,0)
            
            cls_result_cube_30 = model_cube_30.predict(trainX_cube_30)[0][1]
            probability_30_30_30_cube_.append(cls_result_cube_30)


# In[10]:


probability_30_30_30_cube = np.array(probability_30_30_30_cube_)
#probability_30_30_30_cube = probability_30_30_30_cube.clip(0.005,0.995)
#probability_30_30_30_cube = probability_30_30_30_cube.round(3)
test_pred_0['probability'] = probability_30_30_30_cube


# In[11]:


test_pred_0.to_csv(main_path + "evaluationScript/exampleFiles/val_cal_2.csv", index=False)


# In[29]:


ss = pd.read_csv(main_path + "evaluationScript/exampleFiles/val_cal.csv")


# In[30]:


prob = []
for i,j in ss.iterrows():
    if j['diameter_mm'] > 7 and j['diameter_mm'] <10:
        prob.append(j['probability']*0.1)
    else:
        prob.append(j['probability'])
ss['probability'] = prob        


# In[31]:


test_pred_0.to_csv(main_path + "evaluationScript/exampleFiles/submission/val_cal_2.csv", index=False)


# In[32]:
#调用lung16的评分函数文件
import os
os.system('python /devdata1/ding/data/TianChi/ali/evaluationScript/noduleCADEvaluationLUNA16.py')
# get_ipython().magic('run ./evaluationScript/noduleCADEvaluationLUNA16.py')


# In[33]:


# result = plt.imread('/devdata1/ding/data/TianChi/ali/evaluationScript/froc_val_cal_2.png')


# In[34]:


# get_ipython().magic('matplotlib inline')
# plt.figure(num='result',figsize=(16,16))
# plt.imshow(result)


# In[ ]:




