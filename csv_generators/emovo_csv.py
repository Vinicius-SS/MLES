import csv
import sys
import re
import os

with open('emovo.csv', 'w') as csv_file:

	dataset_path = sys.argv[1]

	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(['filename', 'categorical'])

	walker = os.walk(dataset_path)

	emotions = {'dis': 'dis', 'gio': 'hap', 'neu': 'neu', 'pau': 'fea', 
		    'rab': 'ang', 'sor': 'sur', 'tri': 'sad'}

	for d in walker:
		for f in d[2]:
			if f.endswith('.wav'):

				f_path = os.path.join(d[0], f)
				emo = emotions[f[:3]]

				writer.writerow([f_path, emo])
