import csv
import sys
import os
import re

with open('serbian.csv', 'w') as csv_file:

	dataset_path = sys.argv[1]

	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(['filename', 'categorical'])

	walker = os.walk(dataset_path)

	emotions = {'Anger' : 'ang', 'Fear' : 'fea', 'Happiness' : 'hap',
				'Neutral': 'neu', 'Sadness' : 'sad'}

	matcher =  re.compile('|'.join(emotions.keys()))

	for d in walker:
		for f in d[2]:

			if f.endswith('.wav'):

				f_path = os.path.join(d[0], f)
				emo_text = matcher.search(d[0]).group()

				emo = emotions[emo_text]

				writer.writerow([f_path, emo])

