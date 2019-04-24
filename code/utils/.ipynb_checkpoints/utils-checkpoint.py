# -*- coding: utf-8 -*-
import utils.helpers as helpers
from utils.models import *


import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import numpy as np
import pandas as pd
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
import os
import glob
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage
import sklearn
import time

from typing import List, Tuple
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from sklearn.utils import shuffle

from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, BatchNormalization,SpatialDropout2D,Convolution3D,MaxPooling3D, UpSampling3D, Flatten, Dense
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler,EarlyStopping
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import plot_model

random.seed(1321)
numpy.random.seed(1321)

import warnings
warnings.filterwarnings("ignore")

K.set_image_dim_ordering('th') 




def load_train(data_path):
    folders = [x for x in os.listdir(data_path) if 'subset' in x]
    os.chdir(data_path)
    patients = []    
    for i in folders:
        os.chdir(data_path + i)
        patient_ids = [x for x in os.listdir(data_path + i) if '.mhd' in x]
        for id in patient_ids:
            j = '{}/{}'.format(i, id)
            patients.append(j)
    return patients

def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

def cal_dist(var_csv,anno):
    distmin = []
    distmax = []
    ratio = []
    for index_csv,row_csv in var_csv.iterrows():
        mini_ann = anno[anno['seriesuid']==row_csv['seriesuid']] 
        distmin_ = []
        distmax_ = []
        ratio_ = []
        for index_ann,row_ann in mini_ann.iterrows():                   
            a1 = np.absolute(row_csv['coordX']-row_ann['coordX'])
            a2 = np.absolute(row_csv['coordY']-row_ann['coordY'])
            a3 = np.absolute(row_csv['coordZ']-row_ann['coordZ'])
            distmin_.append(np.min(np.array([a1,a2,a3])))    
            distmax_.append(np.max(np.array([a1,a2,a3]))) 
            ratio_.append(np.max(np.array([a1,a2,a3]))/row_ann['diameter_mm'])            
        ratio.append(np.min(ratio_))
        distmin.append(np.min(distmin_))  
        distmax.append(np.min(distmax_)) 
    var_csv['ratio'] = ratio
    var_csv['distmin'] = distmin
    var_csv['distmax'] = distmax
    #var_csv = var_csv.sort_values('distmin')
    return var_csv

def cal_recall(var_csv,anno):
    distmin = []
    distmax = []
    ratio = []
    for index_csv,row_csv in anno.iterrows():
        mini_ann = var_csv[var_csv['seriesuid']==row_csv['seriesuid']] 
        if mini_ann.shape[0] == 0:
            ratio.append(int(2000))
            distmin.append(int(2000)) 
            distmax.append(int(2000)) 
        else:
            distmin_ = []
            distmax_ = []
            ratio_ = []
            #print mini_ann
            for index_ann,row_ann in mini_ann.iterrows():                  
                a1 = np.absolute(row_csv['coordX']-row_ann['coordX'])
                a2 = np.absolute(row_csv['coordY']-row_ann['coordY'])
                a3 = np.absolute(row_csv['coordZ']-row_ann['coordZ'])
                distmin_.append(np.min(np.array([a1,a2,a3])))    
                distmax_.append(np.max(np.array([a1,a2,a3])))  
                ratio_.append(np.max(np.array([a1,a2,a3]))/row_csv['diameter_mm'])
                
            ratio.append(np.min(ratio_))            
            distmin.append(np.min(distmin_))  
            distmax.append(np.min(distmax_)) 
    anno['ratio'] = ratio
    anno['distmin'] = distmin
    anno['distmax'] = distmax
    return anno

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def pred_samples(src,img_file,model):
    patient_id = img_file[:-9]
    img_array = np.load(src + img_file)
    pos_annos = pd.read_csv(src + img_file[:-9] + '_annos_pos.csv')
    origin = np.array([pos_annos.loc[0]['origin_x'],pos_annos.loc[0]['origin_y'],pos_annos.loc[0]['origin_z']]) 
    spacing = np.array([pos_annos.loc[0]['spacing_x'],pos_annos.loc[0]['spacing_y'],pos_annos.loc[0]['spacing_z']]) 
    img_array_new = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):
        img = img_array[i]
        #img = skimage.morphology.binary_closing(np.squeeze(img), np.ones([3,3]))
        seg_img, overlap = helpers.get_segmented_lungs(img.copy())
        overlap = skimage.morphology.binary_opening(np.squeeze(overlap), np.ones([5,5]))
        img = normalize(img) * 255        
        img = np.expand_dims(img,0)
        img = np.expand_dims(img,0)
        p = model.predict(img)
        img_array_new[i] = p * overlap
    img_array_new[img_array_new < 0.5] = 0
    np.save('{}{}{}.npy'.format(src, patient_id,str('_pred')), img_array_new)
    return 

def simule(data_path,model_fenge,lung_100):
    mean = 0.0
    for scan in tqdm(lung_100):
        patient_id = scan.split('/')[-1][:-4]
        img = cv2.imread(data_path + 'lung/' + scan,cv2.IMREAD_GRAYSCALE)   
        #seg_img, overlap = helpers.get_segmented_lungs(img.copy())
        mask = cv2.imread(data_path + 'nodule/' + scan[:-5] + 'm.png',cv2.IMREAD_GRAYSCALE).astype(int) 
        
        
        #img = skimage.morphology.binary_opening(np.squeeze(img), np.ones([2,2])) 
        img = np.expand_dims(img,0)
        img = np.expand_dims(img,0)   
        p = model_fenge.predict(img)

        p = np.squeeze(p)
        


        #细节参数调整
        
        #p = p*overlap
        #p = skimage.morphology.binary_opening(np.squeeze(p), np.ones([5,5]))   
        #seg_img, overlap = helpers.get_segmented_lungs(np.squeeze(img.copy()))
        #overlap = skimage.morphology.binary_opening(np.squeeze(overlap), np.ones([5,5]))
        #p=p*overlap
        
        mean += dice_coef_np(mask,p)
    mean=mean*1.0/len(lung_100)
    print(u"分割的相似度是：%.6f%%"  %(mean*100))
    
def get_coord(img_array_new, patient_id,origin, spacing):
    var = []
#提取连通区域                     
    temp = np.squeeze(img_array_new)


    temp = skimage.morphology.binary_opening(np.squeeze(temp), np.ones([5,5,5]))

    labels = skimage.measure.label(np.squeeze(temp))    
    props = skimage.measure.regionprops(labels)
    for i in range(len(props)):
        #if props[i]['EquivDiameter'] < 100:        
        world_coordinates = voxel_2_world([props[i]['Centroid'][2], 
                                                   props[i]['Centroid'][1], 
                                                   props[i]['Centroid'][0]], origin, spacing)               
        var.append([patient_id,
                    world_coordinates[0],
                    world_coordinates[1],
                    world_coordinates[2],
                    props[i]['EquivDiameter']])        
    return var



def make_mask(center, diam, z, width, height, spacing, origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0] + 5)
    v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
    v_ymin = np.max([0, int(v_center[1] - v_diam) - 5])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    # Convert back to world coordinates for distance calculation
    x_data = [x * spacing[0] + origin[0] for x in range(width)]
    y_data = [x * spacing[1] + origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
    return (mask)


def get_segmented_lungs(im, plot=False):
    # Step 1: Convert into a binary image.
    binary = im < -400
    # Step 2: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)
    # Step 3: Label the image.
    label_image = label(cleared)
    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # Step 6: Closure operation with a disk of radius 10. This operation is    to keep nodules attached to the lung wall.
    selem = disk(10) # CHANGE BACK TO 10
    binary = binary_closing(binary, selem)
    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # Step 8: Superimpose the binary mask on the input image.
    get_high_vals = binary == 0
    im[get_high_vals] = -2000
    return im, binary

def create_samples(data_path,df_node,img_file,pic_path):
    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
        # load the data once
        patient_id = img_file.split('/')[-1][:-4]
        itk_img = SimpleITK.ReadImage(data_path + img_file) 
        img_array = SimpleITK.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():       
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint(np.absolute(center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
            for i, i_z in enumerate(np.arange(int(v_center[2])-1,
                             int(v_center[2])+2).clip(0, num_z-1)): # clip prevents going out of bounds in Z
                img = img_array[i_z]
                seg_img, overlap = helpers.get_segmented_lungs(img.copy())
                img = normalize(img)
                mask = make_mask(center, diam, i_z*spacing[2]+origin[2],
                                 width, height, spacing, origin)
                if img.shape[0] > 512: 
                    print(patient_id)
                else:                    
                    cv2.imwrite(pic_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_i.png',img*255)
                    cv2.imwrite(pic_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_m.png',mask*255)
                    cv2.imwrite(pic_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_o.png',img*mask*255)
    return
                    
def create_tests(df_node,img_file,preded_path):
    patient_id = img_file.split('/')[-1][:-4]
    #读取mhd
    itk_img = SimpleITK.ReadImage(data_path + patient) 
    img_array = SimpleITK.GetArrayFromImage(itk_img) # inde
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())    
    #加上肺部掩膜后进行预测（先加和后加）
    if img_array.shape[1] != 512:
        print(patient_id)
    else:
        img_array_new = np.zeros_like(img_array)
        for i in range(img_array.shape[0]):
            img = img_array[i]
            #seg_img, overlap = helpers.get_segmented_lungs(img.copy())
            img = normalize(img) * 255        
            img = np.expand_dims(img,0)
            img = np.expand_dims(img,0)
            p = model.predict(img)
            img_array_new[i] = p 
    np.save('{}{}{}.npy'.format(preded_path, patient_id,str(_pred)), img_array_new)
    return

def get_sample_from_img(img3d, center_x, center_y, center_z, x_size, z_size):
    start_x = max(center_x - x_size / 2, 0)
    if start_x + x_size > img3d.shape[2]:
        start_x = img3d.shape[2] - bx_size

    start_y = max(center_y - x_size / 2, 0)
    start_z = max(center_z - z_size / 2, 0)
    if start_z + z_size > img3d.shape[0]:
        start_z = img3d.shape[0] - z_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + z_size, start_y:start_y + x_size, start_x:start_x + x_size]
    return res  
    
def create_cls_2d_3d_sample(df_node,img_file,data_path,output_path):
    block_size = 32
    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
        # load the data once
        patient_id = img_file.split('/')[-1][:-4]
        itk_img = SimpleITK.ReadImage(data_path + img_file) 
        img_array = SimpleITK.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():       
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint(np.absolute(center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
            imgs = np.ndarray([3, height, width], dtype=np.float32)
            masks = np.ndarray([3, height, width], dtype=np.uint8)
            for i, i_z in enumerate(np.arange(int(v_center[2]),
                             int(v_center[2])+1).clip(0, num_z-1)): # clip prevents going out of bounds in Z
                img = img_array[i_z]
                img = normalize(img)
                mask = make_mask(center, diam, i_z*spacing[2]+origin[2],
                                 width, height, spacing, origin)
                if img.shape[0] > 512: 
                    print(patient_id)
                    
                masks[i] = mask
                imgs[i] = img_array[i_z]
                np.save(output_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_2d_im.npy', imgs*masks*255)     

            #img_3d_20_6 = get_sample_from_img(img_array, node_x, node_y, node_z, 20, 6)
            #img_3d_30_10 = get_sample_from_img(img_array, node_x, node_y, node_z, 30, 10)
            #img_3d_40_26 = get_sample_from_img(img_array, node_x, node_y, node_z, 40, 26)
            
            #img_3d_20_6 = normalize(img_3d_20_6)
            #img_3d_30_10 = normalize(img_3d_30_10)
            #img_3d_40_26 = normalize(img_3d_40_26)
            
            #np.save(output_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_3d_20_6_i.npy',img_3d_20_6)
            #np.save(output_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_3d_30_10_i.npy',img_3d_30_10)
            #np.save(output_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_3d_i_40_26.npy',img_3d_40_26)
    return
    
def pred_tests(data_path,df_node,img_file,preded_path):
    patient_id = img_file.split('/')[-1][:-4]
    itk_img = SimpleITK.ReadImage(data_path + img_file) 
    img_array = SimpleITK.GetArrayFromImage(itk_img) # inde
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())
    direction = np.array(itk_img.GetDirection())
    pos_annos = []
    if img_array.shape[1] > 512:
        offset_Y = int((img_array.shape[1] - 512)/2)
        offset_Z = int((img_array.shape[2] - 512)/2)
        new_img = np.zeros([img_array.shape[0],512,512])
        new_img[:,:,:] = img_array[:,offset_Y:offset_Y+512,offset_Z:offset_Y+512]
        img_array = new_img.copy()
        origin[1] = origin[1] - offset_Y
        origin[2] = origin[2] - offset_Z
        print(patient_id,str('Large'))
        
    elif img_array.shape[1] < 512:
        new_img = np.zeros([img_array.shape[0],512,512])
        for slice in range(len(img_array.shape[0])):
            new_img[slice] = cv2.resize(img_array[slice], dsize=(512, 512))
        img_array = new_img.copy()
        spacing[1] = spacing[1]*(img_array.shape[1]/512)
        spacing[2] = spacing[2]*(img_array.shape[1]/512)
        print(patient_id,str('Small'))
    np.save('{}{}{}.npy'.format(preded_path, patient_id,str('_orig')), img_array)
    pos_annos.append([patient_id, origin[0], origin[1], origin[2], spacing[0], spacing[1], spacing[2]])
    df_annos = pandas.DataFrame(pos_annos, columns=["seriesuid", "origin_x", "origin_y", "origin_z", 
                                                    "spacing_x", "spacing_y", "spacing_z"])
    df_annos.to_csv(preded_path + patient_id + "_annos_pos.csv", index=False)
    return