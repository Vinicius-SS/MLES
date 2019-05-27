import os
import csv
import sys

with open('emodb.csv', 'w') as csv_file:

	dataset_path = sys.argv[1]

	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(['filename', 'categorical'])

	walker = os.walk(dataset_path)

	emotions = {'W' : 'ang', 'N' : 'neu', 'L' : 'bor', 'E' : 'dis',
		    'A' : 'fea', 'F' : 'hap', 'T' : 'sad' }

	for d in walker:
		for f in d[2]:

			if f.endswith('.wav'):

				f_path = os.path.join(d[0], f)
				emo = emotions[f[5]]

				writer.writerow([f_path, emo])
