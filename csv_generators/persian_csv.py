import csv
import os
import sys

with open('persian.csv', 'w') as csv_file:

	dataset_path = sys.argv[1]

	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(['filename', 'categorical'])

	walker = os.walk(dataset_path)

	emotions = {'1' : 'ang', '2' : 'fea', '3' : 'hap', '4' : 'sad',
		    '5' : 'neu', 'boredom': 'bor', 'Disgust': 'dis', 'Surprise': 'sur'}

	for d in walker:
		for f in d[2]:

			if f.endswith('.wav'):

				f_path = os.path.join(d[0], f)
				emo = emotions[d[0].split('/')[1]]

				writer.writerow([f_path, emo])


