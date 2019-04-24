
# coding: utf-8

# In[1]:
import sys
sys.path.append('..')
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)

from utils.imports import *


# In[2]:


src = PATH['model_test_pred']
model_paths = PATH['model_paths']
model_fenge_path=model_paths + 'final_fenge_VGG.h5'
model = load_model(model_fenge_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
#anno = pd.read_csv(csv_path + 'annotations.csv')


# In[3]:


patients = [x for x in os.listdir(src) if 'orig' in x]


# In[4]:


for patient in tqdm(sorted(patients)):
    pred_samples(src,patient,model)


# In[5]:


patients = [x for x in os.listdir(src) if 'pred' in x]  


# In[6]:


empty0 = pandas.DataFrame({'seriesuid':[],'coordX':[],'coordY':[],'coordZ':[],'diameter_mm':[]})
empty1 = pandas.DataFrame({'seriesuid':[],'coordX':[],'coordY':[],'coordZ':[],'diameter_mm':[]})
empty2 = pandas.DataFrame({'seriesuid':[],'coordX':[],'coordY':[],'coordZ':[],'diameter_mm':[]})

for img_file in tqdm(sorted(patients)):
    patient_id = img_file[:-9]
    img_array = np.load(src + img_file)
    pos_annos = pd.read_csv(src + img_file[:-9] + '_annos_pos.csv')
    origin = np.array([pos_annos.loc[0]['origin_x'],pos_annos.loc[0]['origin_y'],pos_annos.loc[0]['origin_z']]) 
    spacing = np.array([pos_annos.loc[0]['spacing_x'],pos_annos.loc[0]['spacing_y'],pos_annos.loc[0]['spacing_z']])     
    temp = np.squeeze(img_array)

    labels0 = skimage.measure.label(np.squeeze(temp))    
    props0 = skimage.measure.regionprops(labels0)
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
    
    temp1 = skimage.morphology.binary_opening(np.squeeze(temp), np.ones([3,3,3]))
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
        
    temp2 = skimage.morphology.binary_opening(np.squeeze(temp), np.ones([5,5,5]))
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


empty0.to_csv(src + "0final_test_result.csv", index=False)
empty1.to_csv(src + "1final_test_result.csv", index=False)
empty2.to_csv(src + "2final_test_result.csv", index=False)


# In[ ]:




