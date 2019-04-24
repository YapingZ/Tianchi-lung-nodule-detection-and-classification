# -*- coding: utf-8 -*-
import os

#数据主目录
main_path = '/devdata1/ding/data/TianChi/ali/'


PATH = {
#CSV文件目录
'annotations_train' : main_path + 'csv/train/',
'annotations_val' : main_path + 'csv/val/',
'annotations_test' : main_path + 'csv/test/',

#MDH文件目录
'src_train' : main_path + 'Data/Download/train/',
'src_val' : main_path + 'Data/Download/val/',
'src_test' : main_path + 'Data/Download/test/',

#图片存放目录
'pic_train' : main_path + 'Process/train_pic/',
'pic_val' : main_path + 'Process/val_pic/',
#分割模型训练数据
'model_train' : main_path + 'Process/train_mask/',    
'model_val' : main_path + 'Process/val_mask/', 

#送入分割模型的数据存放目录
'model_train_lung' : main_path + 'Process/train_mask/lung/',
'model_train_nodule' : main_path + 'Process/train_mask/nodule/',
    
'model_val_lung' : main_path + 'Process/val_mask/lung/',
'model_val_nodule' : main_path + 'Process/val_mask/nodule/',
    
'model_train_pred' : main_path + 'Process/train_mask/train_pred/',
'model_val_pred' : main_path + 'Process/train_mask/val_pred/',    
'model_test_pred' : main_path + 'Process/train_mask/test_pred/',
    
#送入分类模型的数据存放目录
'cls_path' : main_path + 'Process/train_cls/',   
    

'cls_train_cube_30' : main_path + 'Process/train_cls_cube_36/train/',  
'cls_test_cube_30' : main_path + 'Process/train_cls_cube_36/test/',     
'vgg_cls_train_cube_30_true' : main_path + 'Process/train_cls_cube_36/train/vgg/true', 
'vgg_cls_train_cube_30_false' : main_path + 'Process/train_cls_cube_36/train/vgg/false/', 
'vgg_inception_cls_train_cube_30_true' : main_path + 'Process/train_cls_cube_36/train/vgg_inception/true', 
'vgg_inception_cls_train_cube_30_false' : main_path + 'Process/train_cls_cube_36/train/vgg_inception/false/', 
'vgg_densnet_cls_train_cube_30_true' : main_path + 'Process/train_cls_cube_36/train/vgg_densnet/true', 
'vgg_densnet_cls_train_cube_30_false' : main_path + 'Process/train_cls_cube_36/train/vgg_densnet/false/',
'resnet_densnet_cls_train_cube_30_true' : main_path + 'Process/train_cls_cube_36/train/resnet_densnet/true', 
'resnet_densnet_cls_train_cube_30_false' : main_path + 'Process/train_cls_cube_36/train/resnet_densnet/false/',
'resnet_inception_cls_train_cube_30_true' : main_path + 'Process/train_cls_cube_36/train/resnet_inception/true', 
'resnet_inception_cls_train_cube_30_false' : main_path + 'Process/train_cls_cube_36/train/resnet_inception/false/',
'resnet_cls_train_cube_30_true' : main_path + 'Process/train_cls_cube_36/train/resnet/true', 
'resnet_cls_train_cube_30_false' : main_path + 'Process/train_cls_cube_36/train/resnet/false/', 
'cls_test_cube_30_true' : main_path + 'Process/train_cls_cube_36/test/true/', 
'cls_test_cube_30_false' : main_path + 'Process/train_cls_cube_36/test/false/', 
    
    
#分割、分类模型目录
'model_paths' : main_path + 'Process/model/',
'model_final' : main_path + 'Process/model/final_model/',
}


#检查文件夹，如果没有就新建一个
for i in PATH:
    if not os.path.exists(PATH[i]):
        os.makedirs(PATH[i])
        print(PATH[i],u'maked')

#其他参数
TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 512