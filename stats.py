import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def validation_stats(Ypred, Ytrue, pow=1, **kwargs):

		stat = np.mean(np.power(np.abs(Ypred - Ytrue), pow), axis=0)
		return stat

def accuracy(Ytrue, Ypred):

		assert Ytrue.shape[0] == Ypred.shape[0], 'Predicted and true labels must be equal in number!'

		acc = (Ytrue == Ypred).sum() / Ytrue.shape[0]

		return acc

def binarize_pos_neg(Y):

		mapfun = lambda v: -1 if v <= 3 else 1
		vecfun = np.vectorize(mapfun)
		return vecfun(Y).astype(int)

def ternarize_pos_neu_neg(Y, thresh=0.25):

		mapfun = lambda v: -1 if v <= (3. - thresh) else 1 if v >= (3. + thresh) else 0
		vecfun = np.vectorize(mapfun)
		return vecfun(Y).astype(int)

def plot_confusion_matrix(y_true, y_pred, classes,
						  normalize=False,
						  title=None,
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax
