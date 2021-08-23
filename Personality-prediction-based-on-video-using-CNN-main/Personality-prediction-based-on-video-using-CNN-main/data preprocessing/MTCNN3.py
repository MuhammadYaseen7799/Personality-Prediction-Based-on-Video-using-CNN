# extract and plot each detected face in a photograph
from facenet_pytorch import MTCNN
from cv2 import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import os
from os import path
import tensorflow as tf
from torchvision import models
import torch
from torchvision import transforms
from pathlib import Path

def getface_from_video(path):
    
    # Create face detector
    mtcnn = MTCNN(margin=20, post_process=False)

    # Load a video
    v_cap = cv2.VideoCapture(path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through video, taking a handful of frames to form a batch
    frames = []
    for i in tqdm(range(v_len)):
        
        # Load frame
        success = v_cap.grab()
        if i % 50 == 0:
            success, frame = v_cap.retrieve()
        else:
            continue
        if not success:
            continue
            
        # Add to batch
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    # Detect faces in batch
    try:
        faces = mtcnn(frames)
        for i in range(len(faces)):
            plt.imshow(faces[i].permute(1, 2, 0).int().numpy())
            plt.axis('off')
        #plt.show()
    except:
        print("Error in detection")
    return plt

dir(models)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

TRAIN_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\train\\Extracted\\train videos\\"
TEST_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\test\\Extracted\\test videos\\"
VAL_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\validate\\Extracted\\validation videos\\"
PIC_TRAIN_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\train\\Extracted\\face\\"
PIC_TEST_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\test\\Extracted\\face\\"
PIC_VAL_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\validate\\Extracted\\face\\"

train_videos = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_videos =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
val_videos =  [VAL_DIR+i for i in os.listdir(VAL_DIR)]

i=0
while (i<len(train_videos)):
    if( not (path.exists(os.path.splitext(PIC_TRAIN_DIR+Path(train_videos[i]).name)[0] +".jpg"))):
            print(i)
            fig = getface_from_video(train_videos[i])
            fig.savefig(os.path.splitext(PIC_TRAIN_DIR+Path(train_videos[i]).name)[0] +".jpg", bbox_inches='tight')
    
    i+=1
    
