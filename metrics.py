#!/anaconda3/bin/ python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import re
import time
from scipy import ndimage
from numpy import inf
#from skimage.measure import compare_ssim as ssim # for ubuntu server

from skimage.metrics import structural_similarity as ssim
#from sklearn.metrics import normalized_mutual_info_score as nmi

"""
def dice_coef(img_a, img_b):
    y_true = img_a.reshape(-1)
    y_pred = img_b.reshape(-1)
    intersection = np.sum(np.logical_and(y_true, y_pred))
    y_true = np.sum(y_true)
    y_pred = np.sum(y_pred)
    #intersection = 0
    #sum_y_true = 0
    #sum_y_pred = 0
    #for i in range(len(y_true)):
        #intersection += (y_true[i]*y_pred[i])
        #sum_y_true += y_true[i]
        #sum_y_pred += y_pred[i]
    
    return (2.0 * intersection + 1) / (y_true + y_pred + 1)

def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    
    return (2.0 * intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def total_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.prod(flag, axis=-1)

def rmspe_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square( (y_true - y_pred) / K.clip(K.abs(y_true),K.epsilon(),None) ), axis=-1) )

def SSIM(img_a, img_b, alpha=1.0, beta=1.0, gamma=1.0):
    a = img_a.reshape(-1)
    b = img_b.reshape(-1)
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    sigma_a = var_a**0.5
    sigma_b = var_b**0.5
    sigma_ab = np.cov(a, b, ddof=1)
    K1 , K2, L = 0.01, 0.03, 1.0
    #c1, c2, c3 = (K1*L)**2.0, (K2*L)**2.0, (K2*L)**2.0/2.0
    #c1, c2, c3 = 0.0, 0.0, 0.0
    c1, c2, c3 = 0.0004, 0.0036, 0.0
    #alpha, beta, gamma = 1.0, 1.0, 1.0

    luminance = (2.0*mu_a*mu_b + c1)/(mu_a**2.0 + mu_b**2.0 + c1)
    contrast = (2.0*sigma_a*sigma_b + c2)/(var_a + var_b + c2)
    structure = (sigma_ab[0][1] + c3)/(sigma_a*sigma_b + c3)
    ssim = luminance**alpha * contrast**beta * structure**gamma
    #mssim = np.sum(ssim)/len(ssim)

    return ssim

def ssim_acc(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def ssim_loss(y_true, y_pred):
    return 1.0 - ssim_acc(y_true, y_pred)
"""

def rmse(img_a, img_b):
    err = (np.sum( (img_a - img_b)*(img_a - img_b)/(img_a.shape[0]*img_a.shape[1]) ))**0.5
    return err

def rmspe(img_a, img_b):
    #"""
    with np.errstate(divide='ignore', invalid='ignore'):
        err = (img_a - img_b)/img_a
        err[err==inf] = 0
        err[err==-inf] = 0
        err = np.nan_to_num(err)
        err[err==np.nan] = 0
    #"""
    #err = (img_a - img_b)/(img_a+0.00001)
    err = ( np.sum(err**2)/(img_a.shape[0]*img_a.shape[1]) )**0.5
    return err*100

def mae(img_a, img_b):
    err = np.sum( np.abs(img_a - img_b)/(img_a.shape[0]*img_a.shape[1]) )
    return err

def mse(img_a, img_b):
    err = np.sum( (img_a - img_b)*(img_a - img_b)/(img_a.shape[0]*img_a.shape[1]) )
    return err

def psnr(img_a, img_b):
    err = mse(img_a, img_b)
    return 10 * np.log10((1.0 ** 2) / err)

def snr(img):
    ave = np.mean(img.astype('float64'))
    sd = np.std(img.astype('float64'), ddof=1)
    return 10 * np.log10(ave**2 /sd**2)



def find_outer_rectangle(img):
    cols = np.any(img > 0, axis=1)
    rows = np.any(img > 0, axis=0)
    y_min, y_max = np.where(cols)[0][[0, -1]]
    x_min, x_max = np.where(rows)[0][[0, -1]]
    return y_min, y_max, x_min, x_max

def find_inner_rectangle(img):
    img[img>0] = 1.0
    y_min, y_max, x_min, x_max = find_outer_rectangle(img)
    x_mid, y_mid = int((x_min+x_max)/2), int((y_min+y_max)/2)
    img = ndimage.binary_closing(img, structure=np.ones((16,16)))
    # because img.shape[0]/16 = 16
    min_exist_ratio = 0.0
    min_nn_1 = 0
    for j in range(16, y_mid-y_min):
        for i in range(16, x_mid-x_min):
            temp_img_bbox = img[y_mid-j:y_mid+j, x_mid-i:x_mid+i]
            nn_0 = temp_img_bbox.size - np.count_nonzero(temp_img_bbox)
            nn_1 = np.count_nonzero(temp_img_bbox)
            exist_ratio = (nn_1 - nn_0)/nn_1
            if exist_ratio>=min_exist_ratio and nn_1>min_nn_1:
                min_exist_ratio = exist_ratio
                min_nn_1 = nn_1
                best_i = i
                best_j = j
    return y_mid-best_j, y_mid+best_j, x_mid-best_i, x_mid+best_i

def find_mask_rectangle(img, mask):
    y_min, y_max, x_min, x_max = find_outer_rectangle(img)
    y_mask_min = int((y_min+y_max)/2) - int(mask/2)
    y_mask_max = int((y_min+y_max)/2) + int(mask/2)
    x_mask_min = int((x_min+x_max)/2) - int(mask/2)
    x_mask_max = int((x_min+x_max)/2) + int(mask/2)
    return y_mask_min, y_mask_max, x_mask_min, x_mask_max



def metrics_bbox(metrics, img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_outer_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]#### max + 1 !!!
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    return metrics(img_a_bbox, img_b_bbox)

def metrics_bbox_inner(metrics, img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_inner_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    return metrics(img_a_bbox, img_b_bbox)

def metrics_mid(metrics, img_ref, img_a, img_b, mask):
    y_mask_min, y_mask_max, x_mask_min, x_mask_max = find_mask_rectangle(img_ref, mask)
    ROI_a = img_a[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    ROI_b = img_b[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    return metrics(ROI_a, ROI_b)



def rmse_ex0(img_ref, img_a, img_b):
    temp_err = (img_a - img_b)*(img_a - img_b)
    img_ref[img_ref!=0] = 1
    err = np.sum(temp_err*img_ref)
    nn = np.count_nonzero(img_ref)
    if nn==0: err = 0
    else: err = (err/nn)**0.5
    return err

def rmse_bbox(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_outer_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    rmsebb = rmse(img_a_bbox, img_b_bbox)
    return rmsebb

def rmse_bbox_inner(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_inner_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    rmsebb = rmse(img_a_bbox, img_b_bbox)
    return rmsebb

def rmse_mid(img_ref, img_a, img_b, mask):
    y_mask_min, y_mask_max, x_mask_min, x_mask_max = find_mask_rectangle(img_ref, mask)
    ROI_a = img_a[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    ROI_b = img_b[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    rmsemid = rmse(ROI_a, ROI_b)
    return rmsemid

def rmspe_ex0(img_ref, img_a, img_b):
    with np.errstate(divide='ignore', invalid='ignore'):
        temp_err = ( (img_a - img_b)/(img_a) )**2
        temp_err[temp_err==inf] = 0
        temp_err[temp_err==-inf] = 0
        temp_err = np.nan_to_num(temp_err)
        temp_err[temp_err==np.nan] = 0
    #temp_err = ( (img_a - img_b)/(img_a+0.00001) )**2
    img_ref[img_ref!=0] = 1
    err = np.sum(temp_err*img_ref)
    nn = np.count_nonzero(img_ref)
    if nn==0: err = 0
    else: err = (err/nn)**0.5
    return err*100

def rmspe_bbox(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_outer_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    rmspebb = rmspe(img_a_bbox, img_b_bbox)
    return rmspebb

def rmspe_bbox_inner(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_inner_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    rmspebb = rmspe(img_a_bbox, img_b_bbox)
    return rmspebb

def rmspe_mid(img_ref, img_a, img_b, mask):
    y_mask_min, y_mask_max, x_mask_min, x_mask_max = find_mask_rectangle(img_ref, mask)
    ROI_a = img_a[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    ROI_b = img_b[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    rmspemid = rmspe(ROI_a, ROI_b)
    return rmspemid


def mae_bbox(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_outer_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    maebb = mae(img_a_bbox, img_b_bbox)
    return maebb

def mae_bbox_inner(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_inner_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    maebb = mae(img_a_bbox, img_b_bbox)
    return maebb

def mae_mid(img_ref, img_a, img_b, mask):
    y_mask_min, y_mask_max, x_mask_min, x_mask_max = find_mask_rectangle(img_ref, mask)
    ROI_a = img_a[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    ROI_b = img_b[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    maemid = mae(ROI_a, ROI_b)
    return maemid


def psnr_ex0(img_ref, img_a, img_b):
    temp_err = (img_a - img_b)*(img_a - img_b)
    img_ref[img_ref!=0] = 1
    err = np.sum(temp_err*img_ref)
    nn = np.count_nonzero(img_ref)
    if nn==0: err = 0
    else: err = (err/nn)
    return 10 * np.log10((1.0 ** 2) / err)

def psnr_bbox(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_outer_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    psnrbb = psnr(img_a_bbox, img_b_bbox)
    return psnrbb

def psnr_bbox_inner(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_inner_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    psnrbb = psnr(img_a_bbox, img_b_bbox)
    return psnrbb

def psnr_mid(img_ref, img_a, img_b, mask):
    y_mask_min, y_mask_max, x_mask_min, x_mask_max = find_mask_rectangle(img_ref, mask)
    ROI_a = img_a[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    ROI_b = img_b[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    psnrmid = psnr(ROI_a, ROI_b)
    return psnrmid


def ssim_bbox(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_outer_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    ssimbb = ssim(img_a_bbox, img_b_bbox)
    return ssimbb

def ssim_bbox_inner(img_ref, img_a, img_b):
    y_min, y_max, x_min, x_max = find_inner_rectangle(img_ref)
    img_a_bbox = img_a[y_min:y_max, x_min:x_max]
    img_b_bbox = img_b[y_min:y_max, x_min:x_max]
    ssimbb = ssim(img_a_bbox, img_b_bbox)
    return ssimbb

def ssim_mid(img_ref, img_a, img_b, mask):
    y_mask_min, y_mask_max, x_mask_min, x_mask_max = find_mask_rectangle(img_ref, mask)
    ROI_a = img_a[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    ROI_b = img_b[y_mask_min:y_mask_max, x_mask_min:x_mask_max]
    ssimmid = ssim(ROI_a, ROI_b)
    return ssimmid


def ncc(img_a, img_b):
    norm_a = img_a - np.mean(img_a)
    norm_b = img_b - np.mean(img_b)
    ncc = np.sum(norm_a * norm_b) / np.sqrt(np.sum(norm_a**2)*np.sum(norm_b**2))
    return ncc

def mi(X, Y, bins=32):
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    hist_2d, xedges, yedges = np.histogram2d(X, Y, bins=bins)
    p_xy = hist_2d / float(np.sum(hist_2d))
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_x_y = p_x[:, np.newaxis] * p_y[np.newaxis,:]
    nzs = p_xy > 0
    return np.sum(p_xy[nzs] * np.log2(p_xy[nzs] / p_x_y[nzs]))

def nmi(X, Y, bins=32):
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    hist_2d, xedges, yedges = np.histogram2d(X, Y, bins=bins)
    p_xy = hist_2d / float(np.sum(hist_2d))
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_x_y = p_x[:, np.newaxis] * p_y[np.newaxis,:]
    nzs = p_xy > 0
    return np.sum(p_xy[nzs] * np.log2(p_x_y[nzs])) / np.sum(2 * p_xy[nzs] * np.log2(p_xy[nzs]))

def R2(X, Y):
    return np.sum((X - np.mean(Y))**2) / np.sum((Y - np.mean(Y))**2)
