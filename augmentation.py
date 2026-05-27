#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#from keras.preprocessing.image import ImageDataGenerator
#from scipy.misc import imresize
#from scipy.ndimage.interpolation import rotate
import cv2
import copy

def translation_2d(img, shift):
    outimg = np.roll(img, shift[0], axis=0)
    if shift[0]>=0: outimg[:shift[0],:] = 0
    else: outimg[shift[0]:,:] = 0
    outimg = np.roll(outimg, shift[1], axis=1)
    if shift[1]>=0: outimg[:, :shift[1]] = 0
    else: outimg[:,shift[1]:] = 0
    return outimg

def translation_3d_chfirst(img, shift):
    outimg = np.roll(img, shift[0], axis=1)
    if shift[0]>=0: outimg[:,:shift[0],:] = 0
    else: outimg[:,shift[0]:,:] = 0
    outimg = np.roll(outimg, shift[1], axis=2)
    if shift[1]>=0: outimg[:,:,:shift[1]] = 0
    else: outimg[:,:,shift[1]:] = 0
    return outimg

def translation_3d_chlast(img, shift):
    outimg = np.roll(img, shift[0], axis=0)
    if shift[0]>=0: outimg[:shift[0],:,:] = 0
    else: outimg[shift[0]:,:,:] = 0
    outimg = np.roll(outimg, shift[1], axis=1)
    if shift[1]>=0: outimg[:,:shift[1],:] = 0
    else: outimg[:,shift[1]:,:] = 0
    return outimg

def translation_4d(img, shift):
    outimg = np.roll(img, shift[0], axis=1)
    if shift[0]>=0: outimg[:,:shift[0],:,:] = 0
    else: outimg[:,shift[0]:,:,:] = 0
    outimg = np.roll(outimg, shift[1], axis=2)
    if shift[1]>=0: outimg[:,:,:shift[1],:] = 0
    else: outimg[:,:,shift[1]:,:] = 0
    return outimg

def resize_3d(original_img, shape, rescale_ratio, interpolation='nearest'):
    """
    input : img(depth, height, width), scale = 1.0
    output: outimg (slices, col, row), scale = ratio
    """
    img = pad_512(original_img)
    depth, height, width = img.shape
    slices, col, row = shape
    resize_img = np.zeros(shape, img.dtype)
    outimg = np.zeros(shape, img.dtype)
    def f(val):
        #return int(round(val))
        return int(val)

    if img.shape==shape:
        resize_img = img.copy()
    else:
        for iz in range(slices):
            jz = iz / float(slices/depth)
            
            for iy in range(col):
                jy = iy / float(col/height)
                
                for ix in range(row):
                    jx = ix / float(row/width)
                    
                    if f(jz) < 0 or f(jz) > depth-1 or f(jy) < 0 or f(jy) > height-1 or f(jx) < 0 or f(jx) > width-1:
                        resize_img[iz,iy,ix] = 0
                    else:
                        if interpolation=='nearest':
                            resize_img[iz,iy,ix] = img[f(jz),f(jy),f(jx)]
                        elif interpolation=='bilinear':
                            dz = jz - int(jz)
                            dy = jy - int(jy)
                            dx = jx - int(jx)
                            resize_img[iz,iy,ix] = (1-dz)*(1-dy)*(1-dx)*img[f(jz),f(jy),f(jx)] \
                                + (1-dz)*(1-dy)*dx*img[f(jz),f(jy),f(jx)+1] \
                                + (1-dz)*dy*(1-dx)*img[f(jz),f(jy)+1,f(jx)] \
                                + dz*(1-dy)*(1-dx)*img[f(jz)+1,f(jy),f(jx)] \
                                + (1-dz)*dy*dx*img[f(jz),f(jy)+1,f(jx)+1] \
                                + dz*(1-dy)*dx*img[f(jz)+1,f(jy),f(jx)+1] \
                                + dz*dy*(1-dx)*img[f(jz)+1,f(jy)+1,f(jx)] \
                                + dz*dy*dx*img[f(jz)+1,f(jy)+1,f(jx)+1]

    if rescale_ratio==1.0:
        outimg = resize_img.copy()
    else:
        for iz in range(slices):
            #z = slices/2 - iz -0.5
            #jz = int(slices/2 - 0.5 - z/rescale_ratio)
            jz = int(iz * 1.0)
            
            for iy in range(col):
                y = col/2 - iy - 0.5
                jy = int(col/2 - 0.5 - y/rescale_ratio)
                
                for ix in range(row):
                    x = -row/2 + ix + 0.5
                    jx = int(row/2 - 0.5 + x/rescale_ratio)
                    
                    #if jz < 0 or jz > depth-1 or jy < 0 or jy > height-1 or jx < 0 or jx > width-1:
                    if jz < 0 or jz > slices-1 or jy < 0 or jy > col-1 or jx < 0 or jx > row-1:
                        outimg[iz,iy,ix] = 0
                    else:
                        if interpolation=='nearest':
                            outimg[iz,iy,ix] = resize_img[jz,jy,jx]
                        elif interpolation=='bilinear':
                            dz = jz - int(jz)
                            dy = jy - int(jy)
                            dx = jx - int(jx)
                            outimg[iz,iy,ix] = (1-dz)*(1-dy)*(1-dx)*resize_img[f(jz),f(jy),f(jx)] \
                                + (1-dz)*(1-dy)*dx*resize_img[f(jz),f(jy),f(jx)+1] \
                                + (1-dz)*dy*(1-dx)*resize_img[f(jz),f(jy)+1,f(jx)] \
                                + dz*(1-dy)*(1-dx)*resize_img[f(jz)+1,f(jy),f(jx)] \
                                + (1-dz)*dy*dx*resize_img[f(jz),f(jy)+1,f(jx)+1] \
                                + dz*(1-dy)*dx*resize_img[f(jz)+1,f(jy),f(jx)+1] \
                                + dz*dy*(1-dx)*resize_img[f(jz)+1,f(jy)+1,f(jx)] \
                                + dz*dy*dx*resize_img[f(jz)+1,f(jy)+1,f(jx)+1]
                        
    print('resized from', img.shape, 'to', outimg.shape, 'and rescaled(zoomed) by', rescale_ratio)
    return outimg

def flip_horiz_verti(X_array, Y_array, flipswitch):
    num = X_array.shape[0]
    col = X_array.shape[1]
    row = X_array.shape[2]
    times = 1
    if flipswitch==1: times += 2
    else: times += 1
    temp_X = np.zeros([num*times,col,row,X_array.shape[3]], dtype=np.float32)
    temp_Y = np.zeros([num*times,col,row,Y_array.shape[3]], dtype=np.float32)
    temp_X[0:num,:,:,:] = X_array
    temp_Y[0:num,:,:,:] = Y_array

    if flipswitch!=3:
        temp_X[num:num*2,:,:,:] = X_array[:, ::-1, :, :]#horizontal
        temp_Y[num:num*2,:,:,:] = Y_array[:, ::-1, :, :]
    if flipswitch!=2:
        temp_X[-num:,:,:,:] = X_array[:, :, ::-1, :]#vertical
        temp_Y[-num:,:,:,:] = Y_array[:, :, ::-1, :]
    return temp_X, temp_Y

def rescale(X_array, Y_array, resc_list):
    num = X_array.shape[0]
    col = X_array.shape[1]
    row = X_array.shape[2]
    temp_X = np.zeros([num*(1+len(resc_list)),col,row,X_array.shape[3]], dtype=np.float32)
    temp_Y = np.zeros([num*(1+len(resc_list)),col,row,Y_array.shape[3]], dtype=np.float32)
    temp_X[0:num,:,:,:] = X_array
    temp_Y[0:num,:,:,:] = Y_array
    
    for k, resc in enumerate(resc_list):
        for iy in range(col):
            y = col/2 - iy - 0.5
            jy = int(col/2 - 0.5 - y/resc)
            for ix in range(row):
                x = -row/2 + ix + 0.5
                jx = int(row/2 - 0.5 + x/resc)
                if jx < 0: jx = 0
                if jx > row-1: jx = row-1
                #i = int(iz*isize*isize + iy*isize + ix)
                #j = int(iz*row*col + jy*row + jx)
                if jy < 0 or jy > col-1 or jx < 0 or jx > row-1:
                    temp_X[num*(k+1):num*(k+2),iy,ix,:] = 0
                    temp_Y[num*(k+1):num*(k+2),iy,ix,:] = 0
                else:
                    temp_X[num*(k+1):num*(k+2),iy,ix,:] = X_array[:,jy,jx,:]
                    temp_Y[num*(k+1):num*(k+2),iy,ix,:] = Y_array[:,jy,jx,:]

    return temp_X, temp_Y

def random_rotation(X_array, Y_array, Nangle):
    num = X_array.shape[0]
    col = X_array.shape[1]
    row = X_array.shape[2]
    temp_X = np.zeros([num*(1+Nangle),col,row,X_array.shape[3]], dtype=np.float32)
    temp_Y = np.zeros([num*(1+Nangle),col,row,Y_array.shape[3]], dtype=np.float32)
    temp_X[0:num,:,:,:] = X_array
    temp_Y[0:num,:,:,:] = Y_array
    for i in range(num):
        angles = np.arange(-20, 21)#(0, 180)
        angles = np.delete(angles, 20)
        angles = np.random.choice(angles, size=Nangle, replace=False)
        for j in range(Nangle):
            angle = angles[j]
            affine = cv2.getRotationMatrix2D((round(row/2), round(col/2)), angle, 1.0)
            temp_X_rotate = cv2.warpAffine(X_array[i,:,:,:], affine, (col, row))
            temp_Y_rotate = cv2.warpAffine(Y_array[i,:,:,:], affine, (col, row))
            #temp_X_rotate = rotate(temp_X, angle, reshape=False)
            #temp_Y_rotate = rotate(temp_Y, angle, reshape=False)
            if X_array.shape[3]==1: temp_X[num+i*Nangle+j,:,:,0] = temp_X_rotate
            else: temp_X[num+i*Nangle+j,:,:,:] = temp_X_rotate
            temp_Y[num+i*Nangle+j,:,:,:] = temp_Y_rotate
        #print(temp_X.shape, temp_Y.shape)
    return temp_X, temp_Y

def RandomPatch(X_array, Y_array, Nricap):
    num = X_array.shape[0]
    col = X_array.shape[1]
    row = X_array.shape[2]
    temp_X = np.zeros([num*(1+Nricap),col,row,X_array.shape[3]], dtype=np.float32)
    temp_Y = np.zeros([num*(1+Nricap),col,row,Y_array.shape[3]], dtype=np.float32)
    temp_X[0:num,:,:,:] = X_array
    temp_Y[0:num,:,:,:] = Y_array

    for i in range(num*Nricap):
        w = int(np.round(np.random.normal(row*0.5, 1)))
        h = int(np.round(np.random.normal(col*0.5, 1)))
        wsize = [0, w, 0, w, w, row, w, row]
        hsize = [0, 0, h, h, h, h, col, col]
        cropped_x = {}
        cropped_y = {}
        for k in range(4):
            idx = np.random.randint(0, num)
            cropped_x[k] = X_array[idx, hsize[k]:hsize[k+4], wsize[k]:wsize[k+4], :]
            cropped_y[k] = Y_array[idx, hsize[k]:hsize[k+4], wsize[k]:wsize[k+4], :]
        patched_x = np.concatenate(
            [np.concatenate([cropped_x[0], cropped_x[1]], axis=1), 
             np.concatenate([cropped_x[2], cropped_x[3]], axis=1)], axis=0)
        patched_y = np.concatenate(
            [np.concatenate([cropped_y[0], cropped_y[1]], axis=1), 
             np.concatenate([cropped_y[2], cropped_y[3]], axis=1)], axis=0)
        
        temp_X[num+i,:,:,:] = patched_x
        temp_Y[num+i,:,:,:] = patched_y

    return temp_X, temp_Y


def generate(X_array, Y_array, flip, RICAP, scale, rotate):
    num = X_array.shape[0]
    col = X_array.shape[1]
    row = X_array.shape[2]
    resc_list = [0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.03]
    new_X_array = copy.deepcopy(X_array)
    new_Y_array = copy.deepcopy(Y_array)

    if RICAP>0:
        new_X_array, new_Y_array = RandomPatch(new_X_array, new_Y_array, RICAP)
        #new_X_array_R, new_Y_array_R = RandomPatch(new_X_array, new_Y_array, RICAP)

    if flip>0:
        new_X_array, new_Y_array = flip_horiz_verti(new_X_array, new_Y_array, flip)
        
    if scale>0:
        scaling = []
        interval = int(np.floor(len(resc_list)/scale))
        for i in range(scale):
            scaling.append(resc_list[i*interval])
        new_X_array, new_Y_array = rescale(new_X_array, new_Y_array, scaling)

    if rotate>0:
        new_X_array, new_Y_array = random_rotation(new_X_array, new_Y_array, rotate)
        #new_X_array_r, new_Y_array_r = random_rotation(new_X_array, new_Y_array, rotate)
    """
    ### for RICAP + rotate instead of RICAP x rotate ###
    times = 1
    if flip>0: times += 2
    if flip==2 or flip==3: times +=1
    if RICAP>0: times_R = times * RICAP
    if scale>0: times_s = times * len(resc_list)
    if rotate>0: times_r = times * rotate
    times = times_R + 1 + times_s + 0 + times_r + 1
    new_X_array = np.zeros([num*times, col, row, X_array.shape[3]], dtype=np.float32)
    new_Y_array = np.zeros([num*times, col, row, Y_array.shape[3]], dtype=np.float32)
    new_X_array[0:num*times/2,:,:,:] = new_X_array_R
    new_X_array[num*times/2:,:,:,:] = new_X_array_r
    new_Y_array[0:num*times/2,:,:,:] = new_Y_array_R
    new_Y_array[num*times/2:,:,:,:] = new_Y_array_r
    """
    print(new_X_array.shape, new_Y_array.shape)

    return new_X_array, new_Y_array
