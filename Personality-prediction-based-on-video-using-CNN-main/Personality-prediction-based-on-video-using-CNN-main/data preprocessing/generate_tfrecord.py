from random import shuffle
import glob
import pandas as pd
import tensorflow as tf
import sys
import numpy as np
from PIL import Image 
import cv2
import pickle


def load_image(addr):
    img = np.array(Image.open(addr).resize((231,231), Image.ANTIALIAS))
    img = img.astype(np.uint8)
    return img

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f, encoding='latin1')
        df = pd.DataFrame(pickle_data)
        df.reset_index(inplace=True)
        del df['interview']
        df.columns = ["VideoName","ValueExtraversion","ValueNeuroticism","ValueAgreeableness","ValueConscientiousness","ValueOpenness"]
    return df

########################### for train tf record ########################################

TRAIN_LABEL_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\train\\annotation_training.pkl"
TRAIN_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\train\\Extracted\\face\\"
df = load_pickle(TRAIN_LABEL_DIR)
NUM_VID = len(df)
addrs = []
labels = []
for i in range(NUM_VID):
    filelist=glob.glob(TRAIN_DIR+(df['VideoName'].iloc[i]).split('.mp4')[0]+'.jpg')
    print((df['VideoName'].iloc[i]).split('.mp4')[0]+'.jpg')
    addrs+=filelist
    labels+=[np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)]*100


c = list(zip(addrs, labels))
shuffle(c)
train_addrs, train_labels = zip(*c)
train_filename = 'train_face_full.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.compat.v1.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())


writer.close()
sys.stdout.flush()

print (len(train_addrs), "train images saved.. ")

########################### for validate tf record ########################################



VAL_LABEL_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\validate\\annotation_validation.pkl"
VAL_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\validate\\Extracted\\face\\"
df = load_pickle(VAL_LABEL_DIR)
NUM_VID = len(df)
addrs = []
labels = []
for i in range(NUM_VID):
    filelist=glob.glob(VAL_DIR+(df['VideoName'].iloc[i]).split('.mp4')[0]+'.jpg')
    print((df['VideoName'].iloc[i]).split('.mp4')[0]+'.jpg')
    addrs+=filelist
    labels+=[np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)]*100


c = list(zip(addrs, labels))
shuffle(c)
train_addrs, train_labels = zip(*c)
train_filename = 'validate_face_full.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.compat.v1.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('validate data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'validate/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
               'validate/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())


writer.close()
sys.stdout.flush()

print (len(train_addrs), "validate images saved.. ")


########################### for test tf record ########################################


TEST_LABEL_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\test\\annotation_test.pkl"
TEST_DIR = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\test\\Extracted\\face\\"
df = load_pickle(TEST_LABEL_DIR)
NUM_VID = len(df)
addrs = []
labels = []
for i in range(NUM_VID):
    filelist=glob.glob(TEST_DIR+(df['VideoName'].iloc[i]).split('.mp4')[0]+'.jpg')
    print((df['VideoName'].iloc[i]).split('.mp4')[0]+'.jpg')
    addrs+=filelist
    labels+=[np.array(df.drop(['VideoName'], 1, inplace=False).iloc[i]).astype(np.float32)]*100


c = list(zip(addrs, labels))
shuffle(c)
train_addrs, train_labels = zip(*c)
train_filename = 'test_face_full.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.compat.v1.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('test data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'test/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())


writer.close()
sys.stdout.flush()

print (len(train_addrs), "test images saved.. ")



