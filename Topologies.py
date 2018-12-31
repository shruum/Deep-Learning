import sys
sys.argv = ["python.exe", __file__]
import numpy as np
import pylab
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import os, os.path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.layers.normalization import BatchNormalization

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import tensorflow as tf

import os
import os.path as osp

#from PIL import Image
#import glob
import time as t

global batch_size, nb_epoch, neuron_size, dropout_val, NO_ERROR, X_train, Y_train, x_test, y_test, img_rows, img_cols, size, nb_classes

batch_size = 42
nb_epoch = 500
#img_rows, img_cols = 327, 532
neuron_size = 1024
dropout_val = 0.25
NO_ERROR = 0
X_train =[]
Y_train = []
x_test = []
y_test = []
img_rows, img_cols = 128, 128
# int(128), int(128)
size = img_cols, img_rows
nb_classes = 2
numChannel = 1

def AddSamplesTrain(sample, label):

	global X_train, Y_train, img_rows, img_cols
	#plt.figure(1)
	#plt.show()
	order = (1, img_rows, img_cols, numChannel)
	if K.image_data_format() == 'channels_first':
		order = (1, numChannel, img_rows, img_cols)

	x = np.reshape(np.array(sample, dtype = np.float32), order)
	#plt.imshow(np.asarray(x).squeeze(), cmap='gray')
	#plt.draw()
	#plt.pause(0.001)
	if(len(X_train) == 0):
		X_train = x
	else:
		X_train = np.append(X_train, x, axis = 0)
	Y_train = np.append(Y_train, label)
	return NO_ERROR

def AddSamplesTest(sample, label):
	global x_test, y_test, img_rows, img_cols
	order = (1, img_rows, img_cols, numChannel)

	if K.image_data_format() == 'channels_first':
		order = (1, numChannel, img_rows, img_cols)

	x = np.reshape(np.array(sample, dtype = np.float32), order)

	if(len(x_test) == 0):
		x_test = x
	else:
		x_test = np.append(x_test, x, axis = 0)
	y_test = np.append(y_test, label)
	return NO_ERROR

def SetBatchSize(_batchSize):
	global batch_size
	batch_size =_batchSize
	return NO_ERROR

def SetEpoch(_epoch):
	global nb_epoch
	nb_epoch = _epoch
	return NO_ERROR

def SetImageSize(X, Y):
	global img_rows, img_cols
	img_rows = X
	img_cols = Y
	return NO_ERROR

def Model1(input_shape):
	model = Sequential()

	model.add(Convolution2D(32, kernel_size =(3,3), input_shape=input_shape, border_mode='valid', name="input_node"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Convolution2D(32, (5, 5), border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(32, (5, 5), border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(neuron_size,kernel_initializer='he_normal'))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(neuron_size))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(neuron_size))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	return model

def Model2():
	model = Sequential()

	model.add(Convolution2D(32, 3,3, input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Convolution2D(32, 5, 5, input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(32, 5, 5, input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	#model.add(Convolution2D(32, 5, 5, input_shape=(1, img_rows, img_cols)))
	#model.add(Activation('sigmoid'))
	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(672,init='he_normal'))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(672))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	return model

def Model_LeNet(input_shape):
	model = Sequential()

	model.add(Convolution2D(32, kernel_size =(5,5), input_shape=input_shape, padding='valid', border_mode='valid', name="input_node"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(64, (5, 5), padding='valid', border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	#model.add(Convolution2D(32, (5, 5), border_mode='valid'))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(2,2)))

	#model.add(Convolution2D(32, (5, 5)))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(2,2)))

	#model.add(Dropout(0.25))
	model.add(Flatten())

	model.add(Dense(1024,kernel_initializer='he_normal'))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	#Smodel.add(Dropout(0.25))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	return model

def Model_AlexNet(input_shape):
	model = Sequential()

	model.add(Convolution2D(64, kernel_size =(11,11), strides=4, padding ='valid', input_shape=input_shape, name="input_node"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides = 2))

	model.add(Convolution2D(192, (5, 5), border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides = 2))

	model.add(Convolution2D(384, (3, 3), border_mode='valid'))
	model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(384, (3, 3), border_mode='valid'))
	model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides = 2))
	#model.add(Dropout(0.1))

	model.add(Flatten())

	model.add(Dense(4096,kernel_initializer='he_normal'))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.1))

	model.add(Dense(4096,kernel_initializer='he_normal'))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.1))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	return model

def DefectDetect(input_shape):
	model = Sequential()

	model.add(Convolution2D(32, kernel_size =(3,3), input_shape=input_shape, border_mode='valid', name="input_node"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Convolution2D(32, (5, 5), border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(32, (5, 5), border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	#model.add(Convolution2D(32, (5, 5)))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(1024,kernel_initializer='he_normal'))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	return model

def Train():
	K.set_learning_phase(False)

	global X_train, Y_train, x_test, y_test
	X_train = X_train.astype('float32')
	x_test = x_test.astype('float32')
	X_train /= 255
	x_test /= 255

	Y_train = np_utils.to_categorical(Y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)

	if K.image_data_format() == 'channels_first':
		X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	model = Model_AlexNet(input_shape)

	sgd = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
	adad = keras.optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1e-08)
	model.compile(loss='binary_crossentropy',
				  optimizer='sgd',
				  metrics=['accuracy'])

	for i in range(0,len(model.layers)):
		print(model.layers[i].input_shape)
		print(model.layers[i].name)
	'''
	train_datagen = test_datagen = val_datagen = ImageDataGenerator(
		 rescale = None,
		 rotation_range = 45,  # randomly rotate images in the range (degrees, 0 to 180)
		 width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
		 height_shift_range=0.25,  # randomly shift images vertically (fraction of total height)
		 zoom_range=[0.9,1.1],
		 horizontal_flip=True)

	#train_datagen.fit(X_train)
	#test_datagen.fit(x_test)
	train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)
	test_generator = test_datagen.flow(x_test,y_test,	batch_size=50)
	samplePEpoch = (len(X_train) / batch_size)

	model.fit_generator(
		train_generator,
		steps_per_epoch= samplePEpoch,
		epochs=nb_epoch,
		validation_data=test_generator,
		validation_steps=50)
	'''
	stTime = t.time()

	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
			  verbose=1, validation_data=(x_test, y_test))

	elapsedTimeInMs = (t.time() - stTime)

	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	print('Elapsed Time in (s): ', elapsedTimeInMs)

	print(model.inputs)
	num_output = len(model.outputs)
	print(model.layers)
	output_fld = 'C:/TrainingModel/'
	output_graph_name = 'model.pb'
	prefix_output_node_names_of_final_network = 'output_node'
	pred = [None]*num_output
	pred_node_names = [None]*num_output
	for i in range(num_output):
		pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
		pred[i] = tf.identity(model.output[i], name=pred_node_names[i])
		print(model.output[i])
	print('output nodes names are: ', pred_node_names)

	sess = K.get_session()
	gd = sess.graph.as_graph_def()
	'''
	for node in gd.node:
		if node.op == "Switch":
			node.op = "Identity"
			del node.input[1]

		if node.op == "Shape":
			node.op = "Identity"
		'''
	saver = tf.train.Saver()
	constant_graph = graph_util.convert_variables_to_constants(sess, gd, pred_node_names)
	graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
	saver.save(sess, 'C:/ModelTraining/model.ckpt')
	print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

	nodenames = [n.name for n in sess.graph.as_graph_def().node]

	print(nodenames)
	return NO_ERROR
