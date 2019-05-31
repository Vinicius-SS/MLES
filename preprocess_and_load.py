import vggish_input
import os

import numpy as np
import soundfile as sf
import pandas as pd

def load_dataset(csv_path, dataset_location, **kwargs):

	df = pd.read_csv(csv_path)

	if kwargs['debug']:
		df = df.head(kwargs['debug'])

	n_slices = 0
	X = list()
	Y = list()

	print('Loading dataset...')
	for i, row in df.iterrows():

		f = row['filename']
		#if row['categorical'] in ['xxx', 'bor', 'dis', 'fea', 'sur', 'fru']: continue
		if row['categorical'] in ['xxx']: continue
		y, sr = sf.read(os.path.join(dataset_location, f))

		for factor in kwargs['resamples']:

			new_sr = int(sr * factor)
			yy = vggish_input.waveform_to_examples(y, new_sr)

			for slice in yy:
				X.append(slice)
				Y.append(row['categorical'])

		if (i % (len(df.index) // 100)) == 0:
			print ('{}\t{}/{}'.format(f, i, len(df.index)))

	X = np.array(X)
	Y = np.array(Y)

	return X, Y
