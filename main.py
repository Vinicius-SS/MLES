import argparse
import csv
import os
import code

import numpy as np
import preprocess_and_load as pl

parser = argparse.ArgumentParser()

# command-line arguments
# general
parser.add_argument('--debug', type=int, help='load a limited portion of the datasets', nargs='?', const=500, default=False)
parser.add_argument('--dataset', help='set a path for the dataset. The first argument is a CSV file with training examples\' location and labels. The second argument should point to where the files are.', action='append', nargs=2)

# training hyperparameters
parser.add_argument('--learning_rate', type=float, help='set learning rate for training', default=1e-4)
parser.add_argument('--epochs', type=int, help='set number of epochs for training', default=200)
parser.add_argument('--batch_size', type=int, help='set batch size for training', default=32)
parser.add_argument('--val_split', type=float, help='set size of validation split as a fraction (from 0.0 to 1.0) of the dataset', default=None)
parser.add_argument('--test_split', type=float, help='separate validation and testing data', nargs='?', const=0.1, default=False)
parser.add_argument('--retraining', type=bool, help='set whether to retrain pretrained layers or not', default=False)

# preprocessing (and possibly spectrogram generation) hyperparameters
parser.add_argument('--resamples', type=float, help='Resampling ratios for data augmentation. E.g. 1.0 is the default, 1.2 is 20% faster (higher sampling rate)', default=[1.0], nargs='+')

args = parser.parse_args()
kwargs = vars(args)

datasets = list()

for csv_path, datasets_root in kwargs['dataset']:

	X, Y = pl.load_dataset(csv_path, datasets_root, **kwargs)
	datasets.append((X, Y))

X, Y = zip(*datasets)

# not sure whether this is the fastest way to do this but ok

code.interact(local=locals())

X = np.vstack(X)
Y = np.concatenate(Y)

print(X.shape, Y.shape)
