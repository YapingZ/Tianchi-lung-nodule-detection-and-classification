
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
# ## Train

# In[4]:


src = PATH['model_train_pred']
model_paths = PATH['model_final']
model_fenge_path=model_paths + 'final_fenge_VGG.h5'
model = load_model(model_fenge_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
#anno = pd.read_csv(csv_path + 'annotations.csv')


# ## 生成分割结果npy+csv

# In[5]:


patients = [x for x in os.listdir(src) if 'orig' in x]


# In[6]:


for patient in tqdm(sorted(patients)):
    pred_samples(src,patient,model)


# ## 获取分割结果csv

# In[5]:


patients = [x for x in os.listdir(src) if 'pred.npy' in x]    


# In[6]:

#创建数据表用来保存结节的位置和直径
empty0 = pandas.DataFrame({'seriesuid':[],'coordX':[],'coordY':[],'coordZ':[],'diameter_mm':[]})
empty1 = pandas.DataFrame({'seriesuid':[],'coordX':[],'coordY':[],'coordZ':[],'diameter_mm':[]})
empty2 = pandas.DataFrame({'seriesuid':[],'coordX':[],'coordY':[],'coordZ':[],'diameter_mm':[]})

#遍历结节得到origin和spacing
for img_file in tqdm(sorted(patients)):
    patient_id = img_file[:-9]
    img_array = np.load(src + img_file)
    #img_array[img_array < 0.5] = 0
    pos_annos = pd.read_csv(src + img_file[:-9] + '_annos_pos.csv')
    origin = np.array([pos_annos.loc[0]['origin_x'],pos_annos.loc[0]['origin_y'],pos_annos.loc[0]['origin_z']]) 
    spacing = np.array([pos_annos.loc[0]['spacing_x'],pos_annos.loc[0]['spacing_y'],pos_annos.loc[0]['spacing_z']])     
    temp = np.squeeze(img_array)
    
    #进行坐标转换，将体素坐标转为世界坐标
    labels0 = skimage.measure.label(np.squeeze(temp))  #实现连通区域标记
    props0 = skimage.measure.regionprops(labels0) #返回所有联通区块的列表
    for i in range(len(props0)):     
        if props0[i]['EquivDiameter'] > 3:
            world_coordinates0 = voxel_2_world([props0[i]['Centroid'][2], 
                                                props0[i]['Centroid'][1], 
                                                props0[i]['Centroid'][0]], origin, spacing)
            insertrow0 = pd.DataFrame([[patient_id,
                                      world_coordinates0[0],
                                      world_coordinates0[1],
                                      world_coordinates0[2],
                                      props0[i]['EquivDiameter']]],columns = ['seriesuid','coordX','coordY','coordZ','diameter_mm'])
        
            empty0 = empty0.append(insertrow0,ignore_index=True)

    #使用开运算为3后进行坐标转换
    temp1 = skimage.morphology.opening(np.squeeze(temp), np.ones([3,3,3]))#先腐蚀在膨胀，可消除小物体
    labels1 = skimage.measure.label(np.squeeze(temp1))    
    props1 = skimage.measure.regionprops(labels1)
    
    for i in range(len(props1)):     
        if props1[i]['EquivDiameter'] > 3:
            world_coordinates1 = voxel_2_world([props1[i]['Centroid'][2], 
                                                props1[i]['Centroid'][1], 
                                                props1[i]['Centroid'][0]], origin, spacing)               
        
            insertrow1 = pd.DataFrame([[patient_id,
                                      world_coordinates1[0],
                                      world_coordinates1[1],
                                      world_coordinates1[2],
                                      props1[i]['EquivDiameter']]],columns = ['seriesuid','coordX','coordY','coordZ','diameter_mm'])
        
            empty1 = empty1.append(insertrow1,ignore_index=True)

    #使用为5的开运算后，进行坐标转换
    temp2 = skimage.morphology.opening(np.squeeze(temp), np.ones([5,5,5]))
    labels2 = skimage.measure.label(np.squeeze(temp2))    
    props2 = skimage.measure.regionprops(labels2)
    for i in range(len(props2)):
        if props2[i]['EquivDiameter'] > 3:
            world_coordinates2 = voxel_2_world([props2[i]['Centroid'][2], 
                                                props2[i]['Centroid'][1], 
                                                props2[i]['Centroid'][0]], origin, spacing)               
        
            insertrow2 = pd.DataFrame([[patient_id,
                                      world_coordinates2[0],
                                      world_coordinates2[1],
                                      world_coordinates2[2],
                                      props2[i]['EquivDiameter']]],columns = ['seriesuid','coordX','coordY','coordZ','diameter_mm'])
        
            empty2 = empty2.append(insertrow2,ignore_index=True)
            
empty0 = empty0[['seriesuid','coordX','coordY','coordZ','diameter_mm']]
empty1 = empty1[['seriesuid','coordX','coordY','coordZ','diameter_mm']]
empty2 = empty2[['seriesuid','coordX','coordY','coordZ','diameter_mm']] 


# In[7]:

#生成csv文件，分别是未使用开运算的，使用参数为3的开运算，使用参数为5的开运算
empty0.to_csv(src + "0_vgg_final_result.csv", index=False)
empty1.to_csv(src + "1_vgg_final_result.csv", index=False)
empty2.to_csv(src + "2_vgg_final_result.csv", index=False)


# In[ ]:




