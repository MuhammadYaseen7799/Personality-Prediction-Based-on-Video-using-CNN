from random import shuffle
import matplotlib.pyplot as plt
from PIL import Image as im 
import glob
import pandas as pd
import tensorflow as tf
import sys
import numpy as np
from PIL import Image 
import cv2
import pickle

def load_image_numpy(addr):
        X_data = []
        files = glob.glob (addr+"*.jpg")
        for myFile in files:
            image = cv2.imread (myFile)
            X_data.append (image)

        data = np.array(X_data)
        print('X_data shape:', data.shape)
        
        X_data = data.astype('float32')
        return X_data

def load_pkl(addr):

    with open(addr, 'rb') as f:
        data = pickle.load(f, encoding="bytes") 
    Y_data = np.array(data)
    return Y_data


'''
#############loading images##########################
TRAIN_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\train\\Extracted\\face\\"
X_train = load_image_numpy(TRAIN_DIR)
TEST_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\test\\Extracted\\face\\"
X_test = load_image_numpy(TEST_DIR)
############loading images done######################
'''
############loading labels###########################
TRAIN_LABEL_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\train\\annotation_training.pkl"
Y_train = load_pkl(TRAIN_LABEL_DIR)
TEST_LABEL_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\test\\annotation_test.pkl"
Y_test = load_pkl(TEST_LABEL_DIR)


#print(Y_train[b'openness'][b'USO-o9dHJ3U.005.mp4'])
print(Y_test.shape)
