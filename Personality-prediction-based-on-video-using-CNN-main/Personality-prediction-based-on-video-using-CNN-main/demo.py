import glob
import os
import pickle
import sys
import time
import warnings
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import cv2
from dan import DAN

from tensorflow.keras.models import load_model
from Audioclean import downsample_mono, envelope
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

def FusionofBothModels(Frames,Audio):
    return Frames
def predictFromAudio(args):
    return args

def predictfromAudio(args):

    num = []

    cap = cv2.VideoCapture(args)

    file_name = (args.split(".mp4"))[0]
    ## Creating folder to save all the 100 frames from the video
    print(file_name)
    try:
        os.makedirs("ImageData/testingData/" + file_name)
    except OSError:
        print("Error: Creating directory of data")
        
    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        results.append(y_mean)
        a = np.round(np.mean(np.concatenate(num), axis=0), 3)
        a_json = {
            "Extraversion": a[0],
            "Neuroticism": a[1],
            "Agreeableness": a[2],
            "Conscientiousness": a[3],
            "Openness": a[4],
        }
        return a_json

def predictFromFrames(file_name):

    num = []

    cap = cv2.VideoCapture(file_name)

    file_name = (file_name.split(".mp4"))[0]
    ## Creating folder to save all the 100 frames from the video
    print(file_name)
    try:
        os.makedirs("ImageData/testingData/" + file_name)
    except OSError:
        print("Error: Creating directory of data")

    ## Setting the frame limit to 100
    cap.set(cv2.CAP_PROP_FRAME_COUNT, 101)
    length = 101
    count = 0
    ## Running a loop to each frame and saving it in the created folder
    while cap.isOpened():
        count += 1
        if length == count:
            break
        _, frame = cap.read()
        if frame is None:
            continue

        ## Resizing it to 256*256 to save the disk space and fit into the model
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
        # Saves image of the current frame in jpg file
        name = (
            "ImageData/testingData/" + str(file_name) + "/frame" + str(count) + ".jpg"
        )
        cv2.imwrite(name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    addrs = []

    def load_image(addr):
        img = np.array(Image.open(addr).resize((224, 224), Image.ANTIALIAS))
        img = img.astype(np.uint8)
        return img

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    addrs = []

    filelist = glob.glob("ImageData/testingData/" + str(file_name) + "/*.jpg")
    addrs += filelist

    train_addrs = addrs
    train_filename = "test.tfrecords"  # address to save the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(train_addrs)):
        # Load the image
        img = load_image(train_addrs[i])
        feature = {"test/image": _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    BATCH_SIZE = 20
    REG_PENALTY = 0
    NUM_IMAGES = 100
    N_EPOCHS = 1

    imgs = tf.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=config) as sess:

        model = DAN(imgs, REG_PENALTY=REG_PENALTY, preprocess="vggface")
        tr_reader = tf.TFRecordReader()
        tr_filename_queue = tf.train.string_input_producer(
            ["test.tfrecords"], num_epochs=N_EPOCHS
        )
        _, tr_serialized_example = tr_reader.read(tr_filename_queue)
        tr_feature = {"test/image": tf.FixedLenFeature([], tf.string)}
        tr_features = tf.parse_single_example(
            tr_serialized_example, features=tr_feature
        )

        tr_image = tf.decode_raw(tr_features["test/image"], tf.uint8)
        tr_image = tf.reshape(tr_image, [224, 224, 3])
        tr_images = tf.train.shuffle_batch(
            [tr_image],
            batch_size=BATCH_SIZE,
            capacity=100,
            min_after_dequeue=BATCH_SIZE,
            allow_smaller_final_batch=True,
        )
        init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        file_list = ["param1.pkl", "param2.pkl"]
        epoch = 0
        for pickle_file in file_list:
            error = 0
            model.load_trained_model(pickle_file, sess)
            i = 0
            while i < NUM_IMAGES:
                i += BATCH_SIZE
                try:
                    epoch_x = sess.run(tr_images)
                except:
                    if error >= 5:
                        break
                    error += 1
                    continue
                output = sess.run(
                    [model.output], feed_dict={imgs: epoch_x.astype(np.float32)}
                )
                num.append(output[0])
            epoch += 1
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)

    a = np.round(np.mean(np.concatenate(num), axis=0), 3)
    a_json = {
        "Extraversion": a[0],
        "Neuroticism": a[1],
        "Agreeableness": a[2],
        "Conscientiousness": a[3],
        "Openness": a[4],
    }
    return a_json

def FusionOfBothModels(Frames,Audio):
    VideoAccuracy = 0.8983
    AudioAccuracy = 0.8676
    return (Frames*(VideoAccuracy) + Audio*(AudioAccuracy)) /2

#while (True):
FilePath = "C:/Users/Daniyal/Downloads/react-navbar-v1-master/backend/uploads/testing folder/"
while(os.path.exists(FilePath) == False):
    print('no file found')
    
while(len(os.listdir(FilePath)) == 0):
    print('file not saved yet')
    
outputFromFrames = predictFromFrames(str(os.listdir(FilePath)[0]).split(".mp4")[0])
outputFromAudio = predictFromAudio(str(os.listdir(FilePath)[0]).split(".mp4")[0])
fusedResult = FusionofBothModels(outputFromFrames,outputFromAudio)
text_file = open(FilePath+"result.txt", "w")
n = text_file.write(str(fusedResult))
text_file.close()
print(str(os.listdir(FilePath)[0]).split(".mp4")[0])
