# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import time
from keras.models import Model
from keras.layers import Input, LeakyReLU, BatchNormalization, Activation, Dropout, Concatenate
from keras.layers import Conv2D, UpSampling2D, Conv3D, UpSampling3D, ZeroPadding3D
import h5py
from keras.models import model_from_json
from keras.models import load_model

from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
import tensorflow as tf
import load_image
import augmentation
from metrics import *
from tqdm import tqdm


def Unet2D(input_count, output_count):
    if INPUT_CHANNELS==1: ENERGY = spectral[0]
    elif INPUT_CHANNELS==2: ENERGY = spectral[0] + spectral[1]
    elif INPUT_CHANNELS==3: ENERGY = spectral[0] + spectral[1] + spectral[2]
    else: ENERGY = '4E'
    drop_rate_list = [0.4, 0.3, 0.2, 0.1]

    unet_inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, input_count), name="unet_input")

    num_downsample = int(np.floor(np.log(INPUT_HEIGHT)/np.log(2)))
    list_filter_count = [FILTER_COUNT*min(8, (2**i)) for i in range(num_downsample)]
    
    # Encoder
    enc1 = Conv2D(list_filter_count[0], CONV_FILTER_SIZE, strides=CONV_STRIDE, padding='same', name="unet_"+ENERGY+"_conv2D_1")(unet_inputs)
    list_encoder = [enc1]
    for i, f in enumerate(list_filter_count[1:]):
        name_conv = "unet_" + ENERGY + "_conv2D_" + str(i+2)
        name_bn = "unet_" + ENERGY + "_batchnorm_" + str(i+1)
        enc = encoding_layer(list_encoder[-1], f, name_conv, name_bn)
        list_encoder.append(enc)
    
    # Decoder
    list_filter_count = list_filter_count[:-1][::-1]
    if len(list_filter_count) < num_downsample-1:
        list_filter_count.append(FILTER_COUNT)

    dec1 = decoding_layer(list_encoder[-1], list_encoder[-2], list_filter_count[0], "unet_"+ENERGY+"_upconv2D_1", "unet_"+ENERGY+"_upbatchnorm_1", "unet_"+ENERGY+"_upconv2D2_1", dropout=True, num_drop=drop_rate_list[0])
    list_decoder = [dec1]
    for i, f in enumerate(list_filter_count[1:]):
        name_conv = "unet_" + ENERGY + "_upconv2D_" + str(i+2)
        name_bn = "unet_" + ENERGY + "_upbatchnorm_" + str(i+2)
        name_conv2 = "unet_" + ENERGY + "_upconv2D2_" + str(i+2)
        if i<3:
            d = True
            num_drop = drop_rate_list[i+1]
        else:
            d = False
        dec = decoding_layer(list_decoder[-1], list_encoder[-(i+3)], f, name_conv, name_bn, name_conv2, dropout=d, num_drop=num_drop)
        list_decoder.append(dec)
        
    x = Activation('relu')(list_decoder[-1])
    x = UpSampling2D(UPSAMPLE_SIZE)(x)
    x_mse = Conv2D(output_count, DECONV_FILTER_SIZE, padding='same', name="unet_"+ENERGY+"_upconv2D_last")(x)

    model = Model(inputs=unet_inputs, outputs=x_mse)
    return model

def encoding_layer(x, filter_count, name_conv, name_bn):
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filter_count, CONV_FILTER_SIZE, padding='same', strides=CONV_STRIDE, name=name_conv)(x)
    x = BatchNormalization(name=name_bn)(x)
    return x

def decoding_layer(x, encoded_x, filter_count, name_conv, name_bn, name_conv2, dropout=False, num_drop=0.4):
    x = Activation('relu')(x)
    x = UpSampling2D(UPSAMPLE_SIZE)(x)
    x = Conv2D(filter_count, DECONV_FILTER_SIZE, padding='same', name=name_conv)(x)
    x = BatchNormalization(name=name_bn)(x)
    if dropout: x = Dropout(num_drop)(x)
    if (x.shape[1] != encoded_x.shape[1]):
        x = Conv2D(filter_count, (2,2), padding='valid', strides=1, name=name_conv2)(x)
        x = BatchNormalization()(x)
    x = Concatenate(axis=CONCATENATE_AXIS)([x, encoded_x])
    return x


def train_step(X_train, Y_train, batch_size, LR, file_weight, file_csv, aug_flip, aug_RICAP, aug_scale, aug_rotate):
    #X_train_set, X_val_set, Y_train_set, Y_val_set = load_image.split_train_val(X, Y, num_validation=10)
    #X_train, Y_train = augmentation.generate(X_train_set, Y_train_set, aug_flip, aug_RICAP, aug_scale, aug_rotate)
    #X_val, Y_val = augmentation.generate(X_val_set, Y_val_set, aug_flip, aug_RICAP, aug_scale, aug_rotate)
    cp_cb = ModelCheckpoint(filepath = file_weight, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
    cl_cb = CSVLogger(file_csv, separator=',', append=True)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    train = model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epoch*30, validation_split=0.2, callbacks=[cp_cb, cl_cb, lr_cb])

def train_step_cv(X, Y, batch_size, LR, file_weight, file_csv, aug_flip, aug_RICAP, aug_scale, aug_rotate):
    epochcounter = 0
    last_loss, min_loss = 100000, 100000
    patience = 5
    counter = 0
    for k in range(30):
        X_train_set, X_val_set, Y_train_set, Y_val_set = load_image.split_train_val(X, Y, num_validation=0.2)
        X_train, Y_train = augmentation.generate(X_train_set, Y_train_set, aug_flip, aug_RICAP, aug_scale, aug_rotate)
        X_val, Y_val = augmentation.generate(X_val_set, Y_val_set, aug_flip, aug_RICAP, aug_scale, aug_rotate)

        for o in range(n_epoch):
            if counter >= patience:
                LR = LR * 0.5
                if LR < 0.00001:
                    break
                model.compile(loss='mse', optimizer=Adam(lr=LR), metrics=['accuracy'])
                counter = 0
                patience += 1
                last_loss = 100000
            print(epochcounter, LR, last_loss)
            #cp_cb = ModelCheckpoint(filepath = file_weight, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
            cl_cb = CSVLogger(file_csv, separator=',', append=True)
            #lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
            train = model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, Y_val), callbacks=[cl_cb])
            val_loss = min(train.history['val_loss'])
            if val_loss > last_loss:
                last_loss = val_loss
                counter += 1
            if val_loss < last_loss:
                last_loss = val_loss
                counter = 0
                if epochcounter > 50 and min_loss >= last_loss:
                    min_loss = last_loss
                    model.save_weights(file_weight)
            epochcounter += 1

def plot_history(file_csv, plot_loss, plot_acc, plot_range):
    df = pd.read_csv(file_csv, index_col=[0])
    plt.plot(range(len(df['loss'])), df['loss'], label='train_loss')
    plt.plot(range(len(df['val_loss'])), df['val_loss'], label='val_loss')
    val = np.array(df['val_loss'])
    print(np.argmin(val)+1, len(val))
    plt.scatter(x=np.argmin(val), y=min(val), marker='o', c='k')
    plt.title('model loss')
    plt.ylim(0.0, plot_range)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')
    plt.savefig(plot_loss)
    plt.close('all')

    plt.plot(range(len(df['acc'])), df['acc'], label='train_acc')
    plt.plot(range(len(df['val_acc'])), df['val_acc'], label='val_acc')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.savefig(plot_acc)

def predict(X_test, Y_test, INPUT_ELEMENTS, file_model, file_weight, folder_name, model_name,
            validate, save_img=False):
    #metrics_array = np.zeros([5*4, Y_test.shape[0], INPUT_ELEMENTS])
    
    model = model_from_json(open(file_model).read())
    print('weight file loading...')
    model.load_weights(file_weight)
    print('weight file successfully loaded')
    start_time = time.time()
    predicted = model.predict(X_test)
    print(time.time() - start_time)
    #predicted = np.squeeze(predicted)
    print(predicted.shape)
    for i in range(predicted.shape[0]):
        imgout = predicted[i,:,:,:]
        imgout = np.transpose(imgout, (2, 0, 1))
        imgout.astype('float32').tofile('./predicted_%02d_256x256.raw' % i)
    #functions = [rmse, rmspe, mae, psnr, ssim]


def gaussiannoise0(img_array, sd):
    noise = np.random.normal(0, sd, img_array.shape)
    img_array = img_array + noise
    return img_array

def gaussiannoise0_4d(img, sd):
    outimg = np.zeros(img.shape)
    for c in range(img.shape[3]):
        for k in range(img.shape[0]):
            for i in range(img.shape[1]):
                for j in range(img.shape[2]):
                    outimg[k,i,j,c] = np.random.normal(img[k,i,j,c], sd, 1)
    return outimg

def load_img(filename, datatype, slice_size, col_size, row_size,
             channel_last=False):
    img = np.zeros([slice_size, col_size, row_size], datatype)
    with open(filename) as f:
        f = np.fromfile(f, dtype=datatype, count=slice_size*col_size*row_size)
    img = np.reshape(f, [slice_size, col_size, row_size])
    if channel_last==True:
        img = np.transpose(img, (1,2,0) )
    return img

if __name__ == '__main__':
    np.random.seed(10)
    IMG_HEIGHT, IMG_WIDTH = 512, 512
    INPUT_HEIGHT, INPUT_WIDTH = 256, 256

    
    #spectral = ['80kV', '100kV', '120kV', '6MV'] ########## change!!! switching single/multi ##########
    #spectral = ['80kV', '100kV', '120kV']
    #spectral = ['120kV', '6MV']
    spectral = ['120kV']

    
    element = ['H', 'C', 'N', 'O', 'P', 'Ca']
    INPUT_CHANNELS = len(spectral)
    INPUT_ELEMENTS = len(element)
    FILTER_COUNT = 64
    CONCATENATE_AXIS = -1
    CONV_FILTER_SIZE = 4
    CONV_STRIDE = 2
    UPSAMPLE_SIZE = (2, 2)
    DECONV_FILTER_SIZE = 3


    suffix = "_BHC_noise"# ["", "_BHC", "_noise", "_BHC_noise"]
    model_name = 'Unet_human_120kV_32%s' % (suffix) ########## change!!! ##########

    
    folder_name = '../models/' + model_name
    file_model = './id_SL_' + model_name + '_model_architecture.json'
    file_weight = './id_SL_' + model_name + '_model_weights.hdf5'
    file_csv = folder_name + '/id_SL_' + model_name + '_history.csv'
    plot_loss = folder_name + '/id_SL_' + model_name + '_plot_loss.png'
    plot_acc = folder_name + '/id_SL_' + model_name + '_plot_accuracy.png'

    
    ### load training data ###
    if "BHC" in suffix:
        suffix2 = "_BHC"
    else:
        suffix2 = ""
    """
    X = load_image.load_human('../FBP_human_%s%s' % (spectral[0], suffix2), 'float32', 1, IMG_HEIGHT, IMG_WIDTH, INPUT_HEIGHT, 'train')
    if "noise" in suffix:
        if "kV" in spectral[0]: X = gaussiannoise0_4d(X, 0.025)
        elif "MV" in spectral[0]: X = gaussiannoise0_4d(X, 0.005)
    if INPUT_CHANNELS>1:
        X2 = load_image.load_human('../FBP_human_%s%s' % (spectral[1], suffix2), 'float32', 1, IMG_HEIGHT, IMG_WIDTH, INPUT_HEIGHT, 'train')
        if "noise" in suffix:
            if "kV" in spectral[1]: X2 = gaussiannoise0_4d(X2, 0.025)
            elif "MV" in spectral[1]: X2 = gaussiannoise0_4d(X2, 0.005)
        X = np.concatenate([X, X2], axis=CONCATENATE_AXIS)
    if INPUT_CHANNELS>2:
        X3 = load_image.load_human('../FBP_human_%s%s' % (spectral[2], suffix2), 'float32', 1, IMG_HEIGHT, IMG_WIDTH, INPUT_HEIGHT, 'train')
        if "noise" in suffix:
            if "kV" in spectral[2]: X3 = gaussiannoise0_4d(X3, 0.025)
            elif "MV" in spectral[2]: X3 = gaussiannoise0_4d(X3, 0.005)
        X = np.concatenate([X, X3], axis=CONCATENATE_AXIS)
    if INPUT_CHANNELS>3:
        X4 = load_image.load_human('../FBP_human_%s%s' % (spectral[3], suffix2), 'float32', 1, IMG_HEIGHT, IMG_WIDTH, INPUT_HEIGHT, 'train')
        if "noise" in suffix:
            if "kV" in spectral[3]: X4 = gaussiannoise0_4d(X4, 0.025)
            elif "MV" in spectral[3]: X4 = gaussiannoise0_4d(X4, 0.005)
        X = np.concatenate([X, X4], axis=CONCATENATE_AXIS)
    Y = load_image.load_human('../GroundTruth', 'int16', INPUT_ELEMENTS, IMG_HEIGHT, IMG_WIDTH, INPUT_HEIGHT, 'train')

    ### select model ###
    batch_size = 32
    n_epoch = 50
    LR = 0.001
    if USE_3D==False: model = Unet2D(INPUT_CHANNELS, INPUT_ELEMENTS, USE_indiv)
    if USE_3D==True: model = Unet3D(INPUT_CHANNELS, INPUT_ELEMENTS)
    model.summary()
    #plot_model(model, to_file='%s/id_SL_Unet_model.png' % folder_name,show_shapes=True)
    model.compile(loss='mse', optimizer=Adam(lr=LR), metrics=['accuracy'])
    json_string = model.to_json()
    open(file_model,'w').write(json_string)

    ### train ###
    train_step_cv(X, Y, batch_size, LR, file_weight, file_csv, 0, 0, 0, 0, USE_indiv, USE_3D)########## change!!! ##########
    plot_history(file_csv, plot_loss, plot_acc, 0.0025)
    """
    
    ### load test data ###
    validate = suffix
    if suffix=="_BHC_noise": validate = '_BHCn'
    save_img = True
    X_test = load_image.load_Raw('./testim/', 'float32', 1, IMG_HEIGHT, IMG_WIDTH, INPUT_HEIGHT, 'human')
    #Y_test = load_img('GT_120kV_AF.raw', 'float32', INPUT_ELEMENTS, IMG_HEIGHT, IMG_WIDTH)
    #Y_test = cv2.resize(Y_test, (INPUT_WIDTH, INPUT_HEIGHT))
    Y_test = X_test # for test
    predict(X_test, Y_test, INPUT_ELEMENTS, file_model, file_weight, folder_name, model_name, validate, save_img)
    
    
