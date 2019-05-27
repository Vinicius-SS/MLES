import csv
import sys
import re
import os

with open('ekorpus.csv', 'w') as csv_file:

	dataset_path = sys.argv[1]

	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(['filename', 'categorical'])

	walker = os.listdir(dataset_path)

	emotions = {'joy': 'hap', 'anger': 'ang',
		    'sadness': 'sad', 'neutral': 'neu'}

	matcher = re.compile('|'.join(emotions.keys()))

	for f in walker:
		if f.endswith('.wav'):

			# the emotion is "hidden" somewhere in the text file, which is structured
			# in a JSON-esque scheme. it's easier to just look for the emotion string in it

			name = f.split('.')[0] + '.TextGrid'
			t_name = os.path.join(dataset_path, name)

			full_path = os.path.join(dataset_path, f)

			with open(t_name, 'r') as t:

				emo_text = matcher.search(t.read()).group()
				emo = emotions[emo_text]
				writer.writerow([full_path, emo])
