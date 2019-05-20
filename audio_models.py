import keras
from keras.models import Sequential
from keras.layers import (Convolution2D, MaxPooling2D, Flatten, Dense)
from keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)

def vggish_like(n_outputs):

	model = Sequential()

	model.add(Convolution2D(16, (3, 3), activation='relu', padding='same', name='conv1',
	    input_shape=(96, 64, 1)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

	model.add(Convolution2D(32, (3, 3), activation='relu', padding='same', name='conv2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

	model.add(Convolution2D(32, (3, 3), activation='relu', padding='same', name='conv3/conv3_1'))
	model.add(Convolution2D(32, (3, 3), activation='relu', padding='same', name='conv3/conv3_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

	model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv4/conv4_1'))
	model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv4/conv4_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

	model.add(Flatten())

	model.add(Dense(256, name='fc1'))
	model.add(Dense(n_outputs, name='fc2', activation='softmax'))

	return model

def callbacks():

	checkpoint = ModelCheckpoint('ckpt.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
	early_stopping = EarlyStopping(monitor='val_loss', patience=15)
	lr_reduce = ReduceLROnPlateau(verbose=1, factor=0.5, patience=5)

	return [checkpoint, early_stopping, lr_reduce]
