import argparse
import csv
import os
import code
import keras
import stats

import numpy as np
import preprocess_and_load as pl
import audio_models as models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

# eliminate all entries which do not belong to 'name' dataset
def filter_dataset(X, Y, F, name):

	XYF = zip(X, Y, F)

	filtered = [xyf for xyf in XYF if xyf[2] == name]
	X, Y, F = zip(*filtered)

	return np.array(X), np.array(Y), np.array(F)

parser = argparse.ArgumentParser()

# command-line arguments
# general
parser.add_argument('--debug', type=int, help='load a limited portion of the datasets', nargs='?', const=500, default=False)
parser.add_argument('--dataset', help='set a path for the dataset. The first argument is a CSV file with training examples\' location and labels. The second argument should point to where the files are.', action='append', nargs=2)

# training hyperparameters
parser.add_argument('--learning_rate', type=float, help='set learning rate for training', default=1e-3)
parser.add_argument('--epochs', type=int, help='set number of epochs for training', default=200)
parser.add_argument('--batch_size', type=int, help='set batch size for training', default=32)
parser.add_argument('--val_split', type=float, help='set size of validation split as a fraction (from 0.0 to 1.0) of the dataset', default=0.1)
parser.add_argument('--test_split', type=float, help='separate validation and testing data', default=0.1)
parser.add_argument('--retraining', type=bool, help='set whether to retrain pretrained layers or not', default=False)

# preprocessing (and possibly spectrogram generation) hyperparameters
parser.add_argument('--resamples', type=float, help='Resampling ratios for data augmentation. E.g. 1.0 is the default, 1.2 is 20% faster (higher sampling rate)', default=[1.0], nargs='+')
parser.add_argument('--sr', type=int, help='set sampling rate to load audio', default=16000)

args = parser.parse_args()
kwargs = vars(args)

datasets = list()

for csv_path, datasets_root in kwargs['dataset']:

	X, Y = pl.load_dataset(csv_path, datasets_root, **kwargs)
	name = np.array([csv_path.split('/')[-1][:-4]] * len(Y))

	Y = np.stack((Y, name), axis=1)

	datasets.append((X, Y))

X, Y = zip(*datasets)

# not sure whether this is the fastest way to do this but ok

X = np.vstack(X)
X = np.reshape(X, (*X.shape, 1))
Y = np.concatenate(Y)

lb = LabelEncoder()

unique, counts = np.unique(Y, return_counts=True)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=kwargs['test_split'])

Ytr_onehot = keras.utils.to_categorical(lb.fit_transform(Ytrain[:,0]))
Yts_onehot = keras.utils.to_categorical(lb.fit_transform(Ytest[:,0]))

# model

Ftrain = Ytrain[:,1]
Ftest = Ytest[:,1]
Ytrain = Ytr_onehot
Ytest  = Yts_onehot

model = models.vggish_like(Ytrain.shape[1])

model.summary()

print(X.shape, Y.shape, Ftrain.shape, Ftest.shape)
print(lb.classes_)
print(list(unique), list(counts))

opt = keras.optimizers.Adam(lr=kwargs['learning_rate'])
model.compile(optimizer=opt, loss='categorical_crossentropy')
model.fit(Xtrain, Ytrain, batch_size=kwargs['batch_size'], verbose=1, epochs=200,
			validation_split=kwargs['val_split'], callbacks=models.callbacks())

model.load_weights('ckpt.hdf5')

Ypred = model.predict(Xtest)

Ytest = Ytest.argmax(axis=1)
Ypred = Ypred.argmax(axis=1)

acc = stats.accuracy(Ytest, Ypred)
print(f'Accuracy: {100*acc:.2f}%')
stats.plot_confusion_matrix(Ytest, Ypred, lb.classes_)
plt.show()

code.interact(local=locals())

