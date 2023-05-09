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
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
import numpy as np
from keras.initializers import RandomNormal
from keras.layers import LeakyReLU
#from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from math import sqrt, ceil
from tqdm import tqdm_notebook as tqdm
import cv2
from sklearn.utils import shuffle
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
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()
from glob import glob
from scipy import signal

np.random.seed(42)
create_dir("files")

train_path = "data/kdsb/train/"
valid_path = "data/kdsb/valid/"

    ## Training
train_x = sorted(glob(os.path.join(train_path, "images", "*.png")))
train_y = sorted(glob(os.path.join(train_path, "masks", "*.png")))

    ## Shuffling
train_x, train_y = shuffling(train_x, train_y)
train_x = train_x
train_y = train_y

    ## Validation
valid_x = sorted(glob(os.path.join(valid_path, "images", "*.png")))
valid_y = sorted(glob(os.path.join(valid_path, "masks", "*.png")))


print("final training set length",len(train_x),len(train_y))
print("final valid set length",len(valid_x),len(valid_y))

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

    
X_tot_val = [get_image_new(sample_file,256,256) for sample_file in valid_x]
X_val,edge_x_val = [],[]
print(len(X_tot_val))
for i in range(0,len(valid_x)):
    X_val.append(X_tot_val[i][0])
    edge_x_val.append(X_tot_val[i][1])
X_val = np.array(X_val).astype(np.float32)
edge_x_val = np.array(edge_x_val).astype(np.float32)
edge_x_val  =  np.expand_dims(edge_x_val,axis=3)

if not os.path.exists('edge_detection'):
    os.makedirs('edge_detection')

# Loop through each image in the validation set
for i in range(len(valid_x)):
    # Save the original image
    plt.imsave('edge_detection/{}_original.png'.format(i), X_val[i])
    # Save the edge detection image
    plt.imsave('edge_detection/{}_edge_detection.png'.format(i), edge_x_val[i].squeeze(), cmap='gray')

Y_tot_val = [get_image(sample_file,256,256,gray=True) for sample_file in valid_y]
Y_val,edge_y = [],[]
for i in range(0,len(valid_y)):
    Y_val.append(Y_tot_val[i][0])
Y_val = np.array(Y_val).astype(np.float32)
           
Y_val  =  np.expand_dims(Y_val,axis=3)

def train(epochs, batch_size,output_dir, model_save_dir):
    
    batch_count = int(len(train_x) / batch_size)
    max_val_dice= -1
    G = msrf()
    G.summary()
    optimizer = get_optimizer()
    G.compile(optimizer = optimizer, loss = {'x':seg_loss,'edge_out':'binary_crossentropy','pred4':seg_loss,'pred2':seg_loss},loss_weights={'x':2.,'edge_out':1.,'pred4':1. , 'pred2':1.})
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15,batch_size)
        #sp startpoint
        for sp in range(0,batch_count,1):
            if (sp+1)*batch_size>len(train_x):
                batch_end = len(train_x)
            else:
                batch_end = (sp+1)*batch_size
            X_batch_list = train_x[(sp*batch_size):batch_end]
            Y_batch_list = train_y[(sp*batch_size):batch_end]
            X_tot = [get_image_new(sample_file,256,256) for sample_file in X_batch_list]
            X_batch,edge_x = [],[]
            for i in range(0,batch_size):
                X_batch.append(X_tot[i][0])
                edge_x.append(X_tot[i][1])
            X_batch = np.array(X_batch).astype(np.float32)
            edge_x = np.array(edge_x).astype(np.float32)
            Y_tot = [get_image(sample_file,256,256, gray=True) for sample_file in Y_batch_list]
            Y_batch,edge_y = [],[]
            for i in range(0,batch_size):
                Y_batch.append(Y_tot[i][0])
                edge_y.append(Y_tot[i][1])
            Y_batch = np.array(Y_batch).astype(np.float32)
            edge_y = np.array(edge_y).astype(np.float32)
            Y_batch  =  np.expand_dims(Y_batch,axis=3)
            edge_y = np.expand_dims(edge_y,axis=3)
            edge_x = np.expand_dims(edge_x,axis=3)
            G.train_on_batch([X_batch,edge_x],[Y_batch,edge_y,Y_batch,Y_batch])

        y_pred,_,_,_ = G.predict([X_val,edge_x_val],batch_size=5)
        y_pred = (y_pred >=0.5).astype(int)
        res = mean_dice_coef(Y_val,y_pred)
		
        

        if(res > max_val_dice):
            max_val_dice = res
            G.save('kdsb_ws.h5')
            print('New Val_Dice HighScore',res)    
       
            
model_save_dir = './model/'
output_dir = '/projectnb/cs585bp/rithik/MSRF-Net/output/'
train(5,8,output_dir,model_save_dir)
