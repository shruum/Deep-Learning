import sys
sys.argv = ["python.exe", __file__]
import numpy as np
import pylab
import matplotlib.pyplot as plt
import os, os.path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from PIL import Image
import glob
import time as t

from TopologyDefect import AddSamplesTrain, AddSamplesTest, Train, SetBatchSize, SetImageSize, SetEpoch

batch_size = 20
nb_epoch = 1000
img_rows, img_cols = 227, 227
neuron_size = 1024
dropout_val = 0.25
NO_ERROR = 0
size = img_rows, img_cols

def loadImage(image_list, path, size):
    length=0
    for filename in glob.glob(path): #assuming gif
        im=Image.open(filename)
        im = im.resize(size, Image.ANTIALIAS)
        ar = np.asarray(im, dtype='float32')
        ar = ar.reshape((1,ar.shape[0],ar.shape[1]))
        image_list.append(ar)
        length = length+1
    return length

def LoadImageSamples(goodPath, badPath, size):
    image_list = []
    lengthSamples = loadImage(image_list,goodPath,size)
    X_train = np.array(image_list)
    Y_train = np.zeros(lengthSamples)

    lengthSamples = loadImage(image_list,badPath,size)
    print(badPath)
    X_train = np.array(image_list)
    Y_train_1 = np.ones(lengthSamples)
    Y_train = np.append(Y_train, Y_train_1)
    return  (X_train, Y_train)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imageFolder")
    args = parser.parse_args()
    path = args.imageFolder

    parser = argparse.ArgumentParser()
    parser.add_argument("imageSize")
    args = parser.parse_args()
    image_size = args.imageSize
    (X_train, Y_train) = LoadImageSamples(path+'/Train/Good/*.png', path+'/Train/Bad/*.png', size)
    (x_test, y_test) = LoadImageSamples(path+'/Test/Good/*.png', path+'/Test/Bad/*.png', size)

    SetImageSize(image_size,image_size)
    SetEpoch(1000)
    SetBatchSize(20)

    for s, l in zip(X_train, Y_train):
    	AddSamplesTrain(s,l)

    for s, l in zip(x_test, y_test):
    	AddSamplesTest(s,l)

    Train()

if __name__ == '__main__':
    main()
