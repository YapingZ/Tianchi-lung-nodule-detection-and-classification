
# coding: utf-8

# In[1]:
import sys
sys.path.append('..')

from utils.imports import *


# In[2]:

#生成输入3d图像的坐标
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
    #if res.shape != (32, 32, 32):
        #res = resize(res,[32,32,32],mode='wrap')
    return res

#制作分类的真结节和假阳性结节，生成.npy文件来保存结节特征
def make_cls_true_false(img_array, v_center, d, times, patient_id, node_idx, dst):
    new_x = int(v_center[0])
    new_y = int(v_center[1])
    new_z = int(v_center[2])

    a = 6

    
    if times == 1:
        trainX_1 = get_cube_from_img(img_array, new_x, new_y, new_z, d)

        if trainX_1.shape == (32, 32, 32):
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_oa') + '.npy', trainX_1)
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_ob') + '.npy', np.fliplr(trainX_1))
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_oc') + '.npy', np.flipud(trainX_1))            
    else:
        for j in range(times):
            new_z1 = new_z + random.choice(range(-a,a+1))
            new_y1 = new_y + random.choice(range(-a,a+1))
            new_x1 = new_x + random.choice(range(-a,a+1))
            trainX_2 = get_cube_from_img(img_array, new_x1, new_y1, new_z1, d)
            if trainX_2.shape == (32, 32, 32):
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('a') + '.npy', trainX_2)
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('b') + '.npy', np.fliplr(trainX_2))
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('c') + '.npy', np.flipud(trainX_2))                
    return

#生成分类样本
def create_cls_sample(df_anno,df_pred,img_file,data_path,output_true,output_false):
    mini_df_anno = df_anno[df_anno["file"]==img_file] 
    mini_df_pred = df_pred[df_pred["file"]==img_file]
    if mini_df_anno.shape[0]>0:
        patient_id = img_file[:-9]
        img_array = np.load(data_path + img_file)
        img_array = normalize(img_array)
        pos_annos = pd.read_csv(data_path + img_file[:-9] + '_annos_pos.csv')
        origin = np.array([pos_annos.loc[0]['origin_x'],pos_annos.loc[0]['origin_y'],pos_annos.loc[0]['origin_z']]) 
        spacing = np.array([pos_annos.loc[0]['spacing_x'],pos_annos.loc[0]['spacing_y'],pos_annos.loc[0]['spacing_z']]) 
        
        for node_idx1, cur_row1 in mini_df_anno.iterrows():       
            node_x = cur_row1["coordX"]
            node_y = cur_row1["coordY"]
            node_z = cur_row1["coordZ"]
            diam = cur_row1["diameter_mm"]
            d = int(diam*3+1)
            center = np.array([node_x, node_y, node_z])   
            v_center1 = np.rint(np.absolute(center-origin)/spacing)#对浮点数取整且不改变浮点数类型
            make_cls_true_false(img_array, v_center1, 32, 1, patient_id, node_idx1, output_true)
            make_cls_true_false(img_array, v_center1, 32, 40, patient_id, node_idx1, output_true)
        for node_idx2, cur_row2 in mini_df_pred.iterrows():       
            node_x = cur_row2["coordX"]
            node_y = cur_row2["coordY"]
            node_z = cur_row2["coordZ"]
            diam = cur_row2["diameter_mm"]
            d = int(diam*3+1)
            center = np.array([node_x, node_y, node_z])   
            v_center2 = np.rint(np.absolute(center-origin)/spacing)  
            make_cls_true_false(img_array, v_center2, 32, 1, patient_id, node_idx2, output_false)
            make_cls_true_false(img_array, v_center2, 32, 2, patient_id, node_idx2, output_false)            
    return


# In[3]:


csv_path = PATH['annotations_train']
output_true = PATH['vgg_cls_train_cube_30_true'] 
output_false = PATH['vgg_cls_train_cube_30_false']
pred_csv_path = PATH['model_train_pred']
data_path = PATH['model_train_pred']
anno_csv_new = pd.read_csv(csv_path + "annotations_all.csv")
pred_csv_new = pd.read_csv(pred_csv_path + "0_vgg_pred_csv_new.csv")
#pred_csv_new = pd.read_csv(pred_csv_path + "anno_false_final.csv")


# In[4]:


#pred_csv_new = pred_csv_new[pred_csv_new.index%5 == 0]


# In[5]:


patients = [x for x in os.listdir(data_path) if 'orig.npy' in x]


# In[6]:


anno_csv_new["file"] = anno_csv_new["seriesuid"].map(lambda file_name: get_filename(patients, file_name))
anno_csv_new = anno_csv_new.dropna()
pred_csv_new["file"] = pred_csv_new["seriesuid"].map(lambda file_name: get_filename(patients, file_name))
pred_csv_new = pred_csv_new.dropna()


# In[7]:

#创建进程，全部使用CPU
Parallel(n_jobs=-1)(delayed(create_cls_sample)(anno_csv_new,pred_csv_new,patient,data_path,output_true,output_false) for patient in tqdm(sorted(patients)))

