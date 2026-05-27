#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#from keras.preprocessing.image import ImageDataGenerator
#from scipy.misc import imresize
#from scipy.ndimage.interpolation import rotate
from tqdm import tqdm
import cv2
import os
import os.path
import array
import re

def cutout_image(img_array, size):
    num = img_array.shape[0]
    row = img_array.shape[1]
    col = img_array.shape[2]
    imagemap = np.zeros((num, size, size, img_array.shape[3]))
    row_diff = int(round((row-size)/2))#if 300 to 256, rowdiff = 22
    col_diff = int(round((col-size)/2))
    for i in range(num):
        imagemap[i,:,:,:] = img_array[i, row_diff:int(size+row_diff), col_diff:int(size+col_diff), :]

    return imagemap

def ZeroPadding(img_array, size):
    num = img_array.shape[0]
    row = img_array.shape[1]
    col = img_array.shape[2]
    imagemap = np.zeros([num, size, size, img_array.shape[3]])
    row_diff = int(round((size-row)/2))
    col_diff = int(round((size-col)/2))
    for i in range(num):
        imagemap[i,row_diff:int(row+row_diff), col_diff:int(col+col_diff),:] = img_array[i,:,:,:]

    return imagemap

def resize(img_array, size, interpolation):
    num = img_array.shape[0]
    imagemap =  np.zeros([num, size, size, img_array.shape[3]])
    if img_array.shape[3]==1:
        imagemap = np.squeeze(imagemap, axis=3)
        for i in range(num):
            imagemap[i,:,:] = cv2.resize(img_array[i,:,:,:], (size, size), interpolation)
        imagemap = imagemap[:,:,:,np.newaxis]
    else:
        for i in range(num):
            imagemap[i,:,:,:] = cv2.resize(img_array[i,:,:,:], (size, size), interpolation)

    return imagemap

def resize_maxpool(img, hd, wd):
    dst = np.zeros([hd, wd])
    h, w = img.shape[0], img.shape[1]
    ax = wd / float(w)
    ay = hd / float(h)
    for yd in range(0, hd):
        for xd in range(0, wd):
            x, y = xd/ax, yd/ay
            ox, oy = int(x), int(y)
            if ox > w - 2: ox = w - 2
            if oy > h - 2: oy = h - 2
            #dx = x - ox
            #dy = y - oy
            dst[yd,xd] = max(img[oy,ox], img[oy,ox+1], img[oy+1,ox], img[oy+1,ox+1])
            #dst[yd,xd] = (1 - dx) * (1-dy) * src[oy,ox] + dx * (1-dy) * src[oy,ox+1] + (1-dx) * dy * src[oy,ox+1] + dx * dy * src[oy+1,ox+1]

    return dst

def split_train_val(A, B, num_validation):
    if num_validation >= 1:
        num_val = np.random.choice(range(A.shape[0]), size=num_validation, replace=False)
    else:
        num_val = np.random.choice(range(A.shape[0]), size=int(A.shape[0]*num_validation), replace=False)
        
    val_A = np.zeros([len(num_val), A.shape[1], A.shape[2], A.shape[3]], np.float32)
    val_B = np.zeros([len(num_val), B.shape[1], B.shape[2], B.shape[3]], np.float32)
    for i in range(len(num_val)):
        val_A[i,:,:,:] = A[num_val[i],:,:,:]
        val_B[i,:,:,:] = B[num_val[i],:,:,:]

    train_A = np.delete(A, [num_val], 0)
    train_B = np.delete(B, [num_val], 0)

    return train_A, val_A, train_B, val_B

def normalize_x(image):
    image = image/127
    return image


def load_Simple_Raw_image_2(train_data_image_path, flag, slicesize, colsize, rowsize):
    dcm_list_h=[]
    file_count = 0
    #print(train_data_image_path)
    for name in os.listdir(train_data_image_path):
        _, ext = os.path.splitext(name)
        if (ext == '.dcm'):
            dcm_list_h.append(name)
            file_count += 1
        #print(file_count, name)
        elif (ext == '.raw'):
            dcm_list_h.append(name)
            file_count += 1
    #print(file_count, name)

    #print("No. of files",file_count)
    if flag==0:
        datatype = np.float32
    else:
        datatype = np.int16

    dcm_list_h = sorted(dcm_list_h, key=lambda s: int(re.findall(r'\d+', s)[-1]))
    dcm_list_h = sorted(dcm_list_h, key=lambda s: int(re.findall(r'\d+', s)[-2]))
    dcm_list_h = sorted(dcm_list_h, key=lambda s: int(re.findall(r'\d+', s)[-3]))
    #imgs = np.zeros([file_count, slicesize, colsize, rowsize], datatype)
    imgs = []
    for i in range(file_count):
        openfilename = train_data_image_path + dcm_list_h[i]
        fp = open(openfilename, 'rb')
        f = np.fromfile(fp, dtype=datatype, count=slicesize*colsize*rowsize)
        fp.close()
        imgs = np.append(imgs, f, axis=-1)
    return imgs


def load_Raw_list(folder_path, flag, slicesize, colsize, rowsize):
    image_files = os.listdir(folder_path)
    image_files.sort()
    image = load_Simple_Raw_image_2(folder_path+os.sep, flag, slicesize, colsize, rowsize)
    image = image.reshape([len(image_files), slicesize, colsize, rowsize])

    # flag= 0:train(FBPrecon), 1:regression(material_dens*0.001), 2:classification(material_ratio), 3:segmentation(organID to binary)
    if flag==1:
        image = np.float32(image/1000)#/np.max(f)
    if flag==2:
        image = np.float32(image)
        ref_image = np.zeros([len(image_files), 1, colsize, rowsize], np.float32)
        ref_image = np.sum(image, axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            image = np.true_divide(image, ref_image)
            indexHC = np.where(np.isnan(image[:,0:2,:,:])==True, 0.0, image[:,0:2,:,:])
            indexN = np.where(np.isnan(image[:,2,:,:])==True, 0.70, image[:,2,:,:])
            indexO = np.where(np.isnan(image[:,3,:,:])==True, 0.30, image[:,3,:,:])
            indexPCa = np.where(np.isnan(image[:,4:6,:,:])==True, 0.0, image[:,4:6,:,:])
            indexN = indexN[:,np.newaxis,:,:]
            indexO = indexO[:,np.newaxis,:,:]
            image = np.concatenate([indexHC, indexN, indexO, indexPCa], axis=1)
    #image = np.float32(image.reshape([len(image_files), rowsize, colsize, slicesize]))
    image = np.float32(np.transpose(image, (0,2,3,1) ))
    return image

def load_Raw(folder_path, datatype, slicesize, colsize, rowsize, out_img_size, phantom):
    #folder_path, datatype, slicesize, colsize, rowsize, out_img_size, load_N, mode
    
    filelist = os.listdir(folder_path)
    if phantom=='SL':
        filelist = sorted(filelist, key=lambda s: int(re.findall(r'\d+', s)[-1]))
        filelist = sorted(filelist, key=lambda s: int(re.findall(r'\d+', s)[-2]))
        filelist = sorted(filelist, key=lambda s: int(re.findall(r'\d+', s)[-3]))
    elif phantom=='human':
        filelist = sorted(filelist)# AF_088, AF_091, ..., AM_103
    else:
        print('you have to select phantom type, "SL" or "human". ')
        os.exit(0)
    #print(filelist)
    
    img_array = np.zeros([len(filelist), out_img_size, out_img_size, slicesize], np.float32)
    for idx in range(len(filelist)):
        openfilename = folder_path + os.sep + filelist[idx]
        fp = open(openfilename, 'rb')
        f = np.fromfile(fp, dtype=datatype, count=slicesize*colsize*rowsize)
        fp.close()
        if datatype=='float32':
            temp_img = f.reshape((colsize, rowsize))
            temp_img = cv2.resize(temp_img, (out_img_size, out_img_size))
            img_array[idx,:,:,0] = temp_img
        elif datatype=='int16':
            temp_img = np.float32(f/1000)
            temp_img = temp_img.reshape((slicesize, colsize, rowsize))
            temp_img = np.transpose(temp_img, (1,2,0) )
            temp_img = cv2.resize(temp_img, (out_img_size, out_img_size))
            img_array[idx,:,:,:] = temp_img
    return img_array


def load_ID(folder_path, ID_num, slicesize, colsize, rowsize, out_img_size, phantom, mode):
    filelist = os.listdir(folder_path)
    if phantom=='SL':
        filelist = sorted(filelist, key=lambda s: int(re.findall(r'\d+', s)[-1]))
        filelist = sorted(filelist, key=lambda s: int(re.findall(r'\d+', s)[-2]))
        filelist = sorted(filelist, key=lambda s: int(re.findall(r'\d+', s)[-3]))
        #filelist = sorted(filelist)
        img_array = np.zeros([len(filelist), out_img_size, out_img_size], np.int16)
    elif phantom=='human':
        filelist = sorted(filelist)# AF_088, AF_091, ..., AM_103
        all_slice = 0
        for idx in range(len(filelist)):
            if "AF" in filelist[idx]: temp_slicesize = 346
            elif "AM" in filelist[idx]: temp_slicesize = 220
            all_slice += temp_slicesize
        img_array = np.zeros([all_slice, out_img_size, out_img_size], np.int16)
    else:
        print('you have to select phantom type, SL or human. ')
        os.exit(0)
    #print(filelist)
    
    num = 0
    for idx in range(len(filelist)):
        if phantom=='human' and "AF" in filelist[idx]: slicesize = 348
        elif phantom=='human' and "AM" in filelist[idx]: slicesize = 222
        openfilename = folder_path + os.sep + filelist[idx]
        fp = open(openfilename, 'rb')
        f = np.fromfile(fp, dtype=np.int16, count=slicesize*colsize*rowsize)
        fp.close()
        if phantom=='SL':
            temp_img = f.reshape((colsize, rowsize))
            #temp_img = resize_maxpool(temp_img, (out_img_size, out_img_size))
            img_array[idx,:,:] = temp_img
        elif phantom=='human':
            temp_img = f.reshape((slicesize, colsize, rowsize))
            temp_img = temp_img[1:-1, :, :]# to remove top and bottom slice, which is zero
            #temp_img = np.transpose(temp_img, (1,2,0))
            img_array[num:num+slicesize-2,:,:] = temp_img
            num = num + slicesize - 2
            #for iz in range(temp_img.shape[2]):
                #resized_img = resize_maxpool(temp_img[:,:,iz], out_img_size, out_img_size)
                #img_array[num,:,:] = resized_img
                #num += 1
    
    if mode=='reg':
        img_array = np.expand_dims(img_array, 3)
        out_img_array = np.float32(img_array/(ID_num-1))

    elif mode=='onehot':
        out_img_array = np.zeros([img_array.shape[0], out_img_size, out_img_size, ID_num], np.int16)
        map_onehot = np.identity(ID_num, dtype=np.int16)
        for i in range(img_array.shape[0]):
            out_img_array[i,:,:,:] = map_onehot[img_array[i,:,:]]

    elif mode=='binary':
        map_decimal = np.arange(0, ID_num)
        seg_channel = int(np.ceil(np.log(len(map_decimal))/np.log(2)))
        out_img_array = np.zeros([img_array.shape[0], out_img_size, out_img_size, seg_channel], np.int16)
        map_binary = np.array([np.array([int(x) for x in list(format(ID, '0'+str(seg_channel)+'b'))]) for ID in map_decimal], dtype=np.int16)
        for i in range(img_array.shape[0]):
            out_img_array[i,:,:,:] = map_binary[img_array[i,:,:]]

    else:
        out_img_array = np.expand_dims(img_array, 3)
        
    return out_img_array

def load_human(folder_path, datatype, slicesize, colsize, rowsize, out_img_size, mode, ite_num=0):
    #GroundTruth -> AF_100 -> AF_100_000 -> AF_MD_512x512_000_000.raw
    #kV_reprojectiondata_80kV -> kV_reprojectiondata_80kV_AM_100 -> 80kV_AM_100_099 -> reprojection_float_gpu_099_221.raw
    #FBP_human_80kV -> FBP_human_80kV_AF_100 -> 80kV_AF_100_099 -> FBP_80kV_AF_100_099_347.raw

    filecount = 0
    idx_scale = sorted(os.listdir(folder_path))
    if mode=='AF': idx_scale = idx_scale[0:int(len(idx_scale)/2)]
    elif mode=='AM':
        if int(len(idx_scale)/2)==1:
            idx_scale = [idx_scale[1]]### should be changed ###
        else:
            idx_scale = idx_scale[int(len(idx_scale)/2):]
    for idxxx_ in idx_scale:
        path_scale = folder_path + os.sep + idxxx_
        idx_seed = sorted(os.listdir(path_scale))
        test_folder_num = int(len(idx_seed)*0.2)
        if mode=='train': idx_seed = idx_seed[test_folder_num:]
        elif mode=='test': idx_seed = idx_seed[0:test_folder_num]
        #elif mode=='all': idx_seed = idx_seed[:]
        elif mode=='iterate': idx_seed = [idx_seed[ite_num]]# avoid extracting "1" from "120kV_AF_100_000"

        for idxx_ in idx_seed:
            path_seed = path_scale + os.sep + idxx_# GroundTruth/AF_100/AF_100_000
            idx_raw = sorted(os.listdir(path_seed))
            filecount += len(idx_raw)
            
    img_array = np.zeros([filecount, out_img_size, out_img_size, slicesize], np.float32)
    idx_scale = sorted(os.listdir(folder_path))# AF_088, AF_091, ..., AM_103
    if mode=='AF': idx_scale = idx_scale[0:int(len(idx_scale)/2)]
    elif mode=='AM':
        if int(len(idx_scale)/2)==1:
            idx_scale = [idx_scale[1]]### should be changed ###
        else:
            idx_scale = idx_scale[int(len(idx_scale)/2):]
    print(idx_scale)
    idx = 0
    for idxxx_ in idx_scale:
        path_scale = folder_path + os.sep + idxxx_# GroundTruth/AF_100
        idx_seed = sorted(os.listdir(path_scale))# AF_100_000, AF_100_001, ..., AF_100_099
        test_folder_num = int(len(idx_seed)*0.2)
        if mode=='train': idx_seed = idx_seed[test_folder_num:]
        elif mode=='test': idx_seed = idx_seed[0:test_folder_num]
        #elif mode=='all': idx_seed = idx_seed[:]
        elif mode=='iterate': idx_seed = [idx_seed[ite_num]]
        #if mode=='train': idx_seed = idx_seed[0:-test_folder_num]
        #if mode=='test': idx_seed = idx_seed[-test_folder_num:]
        print(idx_seed)

        for idxx_ in tqdm(idx_seed, total=len(idx_seed)):
            path_seed = path_scale + os.sep + idxx_# GroundTruth/AF_100/AF_100_000
            idx_raw = sorted(os.listdir(path_seed))

            for idx_ in idx_raw:
                openfilename = path_seed + os.sep + idx_
                fp = open(openfilename, 'rb')
                f = np.fromfile(fp, dtype=datatype, count=slicesize*colsize*rowsize)
                fp.close()
                if slicesize==1:
                    temp_img = f.reshape((colsize, rowsize))
                    temp_img = cv2.resize(temp_img, (out_img_size, out_img_size))
                    img_array[idx,:,:,0] = temp_img
                else:
                    temp_img = np.float32(f/1000)
                    temp_img = temp_img.reshape((slicesize, colsize, rowsize))
                    temp_img = np.transpose(temp_img, (1,2,0) )
                    temp_img = cv2.resize(temp_img, (out_img_size, out_img_size))
                    img_array[idx,:,:,:] = temp_img
                idx += 1

    #train_array = img_array[0:-test_key,:,:,:]
    #test_array = img_array[-test_key:,:,:,:]
    #return train_array, test_array
    return img_array

