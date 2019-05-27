import csv
import sys
import os

with open('spanish.csv', 'w') as csv_file:

	dataset_path = sys.argv[1]

	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(['filename', 'categorical'])

	emotions = {'a' : 'ang', 't' : 'sad', 'j' : 'hap', 'f' : 'fea',
				'd' : 'dis', 's' : 'sur', 'n' : 'neu', 'l' : 'neu',
				'h' : 'neu', 'w' : 'neu', 'z' : 'neu'}

	walker = os.walk(dataset_path)

	for d in walker:
			for f in d[2]:

				if f.endswith('.wav'):

					f_path = os.path.join(d[0], f)
					emo_key = d[0][-4]

					emo = emotions[emo_key]
					writer.writerow([f_path, emo])

