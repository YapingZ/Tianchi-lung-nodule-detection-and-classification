# -*- coding: utf-8 -*-
import os

#数据主目录
main_path = '/media/solo/Data/Datasets/Tianchi/'


PATH = {
#CSV文件目录
'annotations_train' : main_path + 'csv/train/',
'annotations_val' : main_path + 'csv/val/',
'annotations_test' : main_path + 'csv/test/',

#MDH文件目录
'src_train' : main_path + 'stage1/train/',
'src_val' : main_path + 'stage1/val/',
'src_test' : main_path + 'stage1/test/',

#图片存放目录
'pic_train' : main_path + 'tmp/train_pic/',
'pic_val' : main_path + 'tmp/val_pic/',
#分割模型训练数据
'model_train' : main_path + 'tmp/train_mask/',    
'model_val' : main_path + 'tmp/val_mask/', 

#送入分割模型的数据存放目录
'model_train_lung' : main_path + 'tmp/train_mask/lung/',
'model_train_nodule' : main_path + 'tmp/train_mask/nodule/',
    
'model_val_lung' : main_path + 'tmp/val_mask/lung/',
'model_val_nodule' : main_path + 'tmp/val_mask/nodule/',
    
'model_train_pred' : main_path + 'tmp/train_mask/train_pred/',
'model_val_pred' : main_path + 'tmp/train_mask/val_pred/',    
'model_test_pred' : main_path + 'tmp/train_mask/test_pred/',
    
#送入分类模型的数据存放目录
'cls_path' : main_path + 'tmp/train_cls/',   
    

'cls_train_cube_30' : main_path + 'tmp/train_cls_cube_36/train/',  
'cls_test_cube_30' : main_path + 'tmp/train_cls_cube_36/test/',     
'cls_train_cube_30_true' : main_path + 'tmp/train_cls_cube_36/train/true/', 
'cls_train_cube_30_false' : main_path + 'tmp/train_cls_cube_36/train/false/',      
'cls_test_cube_30_true' : main_path + 'tmp/train_cls_cube_36/test/true/', 
'cls_test_cube_30_false' : main_path + 'tmp/train_cls_cube_36/test/false/', 
    
    
#分割、分类模型目录
'model_paths' : main_path + 'tmp/model/',
'model_final' : main_path + 'tmp/model/final_model/',
}


#检查文件夹，如果没有就新建一个
for i in PATH:
    if not os.path.exists(PATH[i]):
        os.mkdir(PATH[i])
        print(PATH[i],u'maked')

#其他参数
TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 512