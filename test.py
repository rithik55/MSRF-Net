import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
np.random.seed(123)
import itertools
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D
from keras import backend as K
from keras.layers import concatenate ,Lambda
import itertools
from tensorflow.keras.layers import BatchNormalization
from keras.optimizers import SGD
import tensorflow as tf
from keras.optimizers import Adam,RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
import numpy as np
from keras.initializers import RandomNormal
from keras.layers import LeakyReLU
from keras.models import Model
from math import sqrt, ceil
from tqdm import tqdm_notebook as tqdm
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import tifffile as tif
from model import msrf
from model import *
from tensorflow.keras.callbacks import *
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
from loss import *
from utils import *
from loss import *
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()
from scipy import signal
from glob import glob


test_img_list = glob("data/kdsb/test/images/*.png")
test_mask_list = glob("data/kdsb/test/masks/*.png")


G = msrf()
G.load_weights('kdsb_ws.h5')
G.summary()
optimizer = get_optimizer()
G.compile(optimizer = optimizer, loss = {'x':seg_loss,'edge_out':'binary_crossentropy','pred4':seg_loss,'pred2':seg_loss},loss_weights={'x':1.,'edge_out':1.,'pred4':1. , 'pred2':1.})



import random
def non_max(I):
    I = np.array(I)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = I.astype(np.float32)/255.
    I= cv2.GaussianBlur(I,(3,3),1.8)
    dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same',
    boundary='symm')
    dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same',
    boundary='symm')
    mag = np.sqrt(dx**2 + dy**2) #mag = normalize(mag)
    angle = np.arctan2(dy, dx) #getting the angle of the edge direction angle = np.rad2deg(angle) #convert to degrees
    non_max_mag = np.zeros(mag.shape) 
    for i in range(1, mag.shape[0] - 1):
        for j in range(1, mag.shape[1] - 1):#iterating through the image
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180): #if the direction is between the given angles we have to check horizontally, i.e, j+1 and j-1
                max_mag = max(mag[i, j - 1], mag[i, j + 1])
            elif (22.5 <= angle[i, j] < 67.5): #check the positive diagonally alligned neigbours
                max_mag = max(mag[i - 1, j - 1], mag[i + 1, j + 1])
            elif (67.5 <= angle[i, j] < 112.5): #check the vertically alligned neigbours
                max_mag = max(mag[i - 1, j], mag[i + 1, j]) 
            else:
                max_mag = max(mag[i + 1, j - 1], mag[i - 1, j + 1]) #negative diagonally alligned neigbours
            if mag[i, j] >= max_mag:
                non_max_mag[i, j] = mag[i, j] #supresing the pixels
    non_max_mag = non_max_mag / 1.5
    non_max_mag = non_max_mag * 255.
    non_max_mag = np.clip(non_max_mag, 0, 255) 
    non_max_mag = non_max_mag.astype(np.float32)
    return non_max_mag

def get_image_new(image_path, image_size_width, image_size_height,gray=False):
    # load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
       
    if gray==True:
        img = img.convert('L')
    # center crop
    img_center_crop = img
    # resize
    img = img.resize((256,256))
    edge = non_max(img)
    #img_resized = img.resize((256, 256))
    #print("resized img:",img_resized.shape)
    #edge = cv2.Canny(np.asarray(np.uint8(img_resized)),10,1000)
    
    flag = False
    # convert to numpy and normalize
    img_array = np.asarray(img).astype(np.float32)/255.0
    #edge = np.asarray(edge).astype(np.float32)/255.0
    #print(img_array)
    if gray==True:
        img_array=(img_array >=0.5).astype(int)
    img.close()
    return img_array,edge



X_tot_test = [get_image_new(sample_file,256,256) for sample_file in test_img_list]
X_test,edge_x_test = [],[]
for i in range(0,len(test_img_list)):
    X_test.append(X_tot_test[i][0])
    edge_x_test.append(X_tot_test[i][1])
X_test = np.array(X_test).astype(np.float32)
edge_x_test = np.array(edge_x_test).astype(np.float32)
print(edge_x_test.shape)
edge_x_test  =  np.expand_dims(edge_x_test,axis=3)
Y_tot_test = [get_image(sample_file,256,256,gray=True) for sample_file in test_mask_list]
Y_test,edge_y_test = [],[]
for i in range(0,len(test_img_list)):
    Y_test.append(Y_tot_test[i][0])
Y_test = np.array(Y_test).astype(np.float32)
           
Y_test  =  np.expand_dims(Y_test,axis=3)


y_pred,_,_,_ = G.predict([X_test,edge_x_test],batch_size=5)
y_pred = (y_pred >=0.5).astype(int)
res = mean_dice_coef(Y_test,y_pred)
print("dice coef on test set",res)

def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_true * y_pred).sum()

    #intersection = np.sum(intersection)   
    union = y_true.sum() + y_pred.sum() - intersection
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (intersection + 1e-15) / (union + 1e-15),tp/(tp+fp),tp/(tp+fn)

res = compute_iou(y_pred,Y_test)
print('iou on test set is ',res[0]," precision is ",res[1]," recall is ",res[2])
