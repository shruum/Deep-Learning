import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras as k

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import os, os.path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

from PIL import Image
import glob
import time as t

from tkinter import filedialog
import tensorflow as tf

img_rows, img_cols = int(64), int(64)

def loadImage(image_list, path, size):
	length=0
	#plt.figure(1)
	#plt.show()
	for filename in glob.glob(path): #assuming gif
		im=Image.open(filename)
		im = im.resize(size, Image.ANTIALIAS)
		ar = np.asarray(im, dtype='float32')
		ar = ar.reshape((1,ar.shape[0],ar.shape[1]))
		#plt.imshow(np.asarray(ar).squeeze(), cmap='gray')
		#plt.draw()
		#plt.pause(0.001)
		image_list.append(ar)
		length = length+1
	return length

def loadImageAndFileList(image_list, file_list, path, size):
	length=0
	#plt.figure(1)
	#plt.show()
	for filename in glob.glob(path): #assuming gif
		im=Image.open(filename)
		im = im.resize(size, Image.ANTIALIAS)
		ar = np.asarray(im, dtype='float32')
		ar = ar.reshape((1,ar.shape[0],ar.shape[1]))
		#plt.imshow(np.asarray(ar).squeeze(), cmap='gray')
		#plt.draw()
		#plt.pause(0.001)
		image_list.append(ar)
		file_list.append(filename)
		length = length+1
	return length():

def loadImageSingle(path, size):
	im=Image.open(path)
	im = im.resize(size, Image.ANTIALIAS)
	ar = np.asarray(im, dtype='float32')
	ar = ar.reshape((1,1,ar.shape[0],ar.shape[1]))
	if K.image_data_format() == 'channels_first':
		x_test = ar.reshape(ar.shape[0], 1, img_rows, img_cols)
	else:
		x_test = ar.reshape(ar.shape[0], img_rows, img_cols, 1)
	return x_test

def LoadImageSamples(goodPath, badPath, size):
	image_list = []
	lengthSamples = loadImage(image_list,goodPath,size)
	X_train = np.array(image_list)
	Y_train = np.zeros(lengthSamples)

	lengthSamples = loadImage(image_list,badPath,size)
	X_train = np.array(image_list)
	Y_train_1 = np.ones(lengthSamples)
	Y_train = np.append(Y_train, Y_train_1)
	return  (X_train, Y_train)

def getVersion(version):
	version = sys.version

def main_tf_predict_multiple(image_path, model):
  # Import data
	size = img_cols, img_rows
	(x_test, y_test) = LoadImageSamples(image_path+'/Test/Good/*.png', image_path+'/Test/Bad/*.png', size)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	size = img_cols, img_rows
	nb_classes = 2
	y_test = np_utils.to_categorical(y_test, nb_classes)
	x_test = x_test.astype('float32')
	x_test /= 255

	y = tf.placeholder(tf.float32, [None, nb_classes])
	y_ = tf.placeholder(tf.float32, [None, nb_classes])

	with tf.Session() as sess:
		with open(model, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, name='')

		stTime = t.time()

		out = sess.graph.get_tensor_by_name('output_node0:0')
		all_predictions = np.zeros(shape=(len(x_test),nb_classes))
		for i in range(len(x_test)):
			x_test_t = x_test[i]
			x_test_t = x_test_t.reshape(1, img_rows, img_cols, 1)
			predictions = sess.run(out, {'input_node_input:0': x_test_t})
			all_predictions[i,:] = np.squeeze(predictions)

		elapsedTimeInMs = (t.time() - stTime) * 1000
		print(len(x_test))
		timeString = str(elapsedTimeInMs/len(x_test))
		print('Elapsed Time in (ms): ', timeString)
		#print(all_predictions)
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		result = sess.run(accuracy, feed_dict={y: all_predictions, y_: y_test})
		print('Accuracy: ', result)

def main_tf_predict_single():
  # Import data
	string = 'g'

	while string != 'n':
		size = img_cols, img_rows
		nb_classes = 2
		file_path_string = filedialog.askopenfilename()
		x_test = loadImageSingle(file_path_string, size)
		x_test = x_test.astype('float32')
		x_test /= 255

		y = tf.placeholder(tf.float32, [None, nb_classes])
		y_ = tf.placeholder(tf.float32, [None, nb_classes])

		model_path_string = filedialog.askopenfilename()
		with tf.Session() as sess:
			with open(model_path_string, 'rb') as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				tf.import_graph_def(graph_def, name='')

			stTime = t.time()

			out = sess.graph.get_tensor_by_name('output_node0:0')
			predications = sess.run(out, {'input_node_input:0': x_test})

			elapsedTimeInMs = (t.time() - stTime) * 1000
			timeString = str(elapsedTimeInMs)

		#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		#result = sess.run(accuracy, feed_dict={y: predications,
		#									   y_: mnist.test.labels})
		print(predications)
		print(elapsedTimeInMs)

		clr ='green'
		passstring = 'Good Sample'
		if np.argmax(predications) == 1:
			passstring = 'Bad Sample'
			clr='red'

		passstring = passstring + '\n\n\n' + ' Elapsed Time ' + timeString +' ms'

		fig = plt.figure(1)
		fig.suptitle(passstring,color=clr, fontsize=14, fontweight='bold')
		plt.imshow(np.asarray(x_test).squeeze(), cmap='gray')
		plt.draw()
		plt.pause(0.001)
		plt.show(block=False)
		string = input("Do you want to continue? y/n ")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagePath")
    args = parser.parse_args()
    image_path = args.imagePath

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    model = args.model
	#main_tf_predict_single()
	main_tf_predict_multiple(image_path, model)

if __name__ == '__main__':
    main()
