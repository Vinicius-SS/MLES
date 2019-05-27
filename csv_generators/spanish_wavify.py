import librosa
import os
import sys
import struct

import numpy as np

def wavify(f_path):

	with open(f_path, 'rb') as ff:

		yb = ff.read()
		y = struct.unpack('h' * (len(yb) // 2), yb)
		y = np.array(y, dtype=np.float32)

		f_base = '.'.join(f_path.split('.')[:-1])
		f_wav = f_base + '.wav'

		print('Writing {}'.format(f_wav))
		librosa.output.write_wav(f_wav, y / 1., 16000, norm=True)

dataset_path = sys.argv[1]
walker = os.walk(dataset_path)

for d in walker:
	for f in d[2]:

		if f.endswith('.l16'):

			f_path = os.path.join(d[0], f)
			wavify(f_path)
