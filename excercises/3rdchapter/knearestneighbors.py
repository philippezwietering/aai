import numpy as np
import math
from statistics import *

data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5:lambda x: 0 if x == b'-1' else float(x), 7:lambda x: 0 if x == '-1' else float(x)})

dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
labels = []
for label in dates:
	if label < 20000301:
		labels.append('winter')
	elif 20000301 <= label < 20000601:
		labels.append('lente')
	elif 20000601 <= label < 20000901:
		labels.append('zomer')
	elif 20000901 <= label < 20001201:
		labels.append('herfst')
	else:
		labels.append('winter')
#print(data)
#print(labels)

valdata = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5:lambda x: 0 if x == b'-1' else float(x), 7:lambda x: 0 if x == '-1' else float(x)})

valdates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
vallabels = []
for label in valdates:
	if label < 20010301:
		vallabels.append('winter')
	elif 20010301 <= label < 20010601:
		vallabels.append('lente')
	elif 20010601 <= label < 20010901:
		vallabels.append('zomer')
	elif 20010901 <= label < 20011201:
		vallabels.append('herfst')
	else:
		vallabels.append('winter')

daysdata = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5:lambda x: 0 if x == b'-1' else float(x), 7:lambda x: 0 if x == '-1' else float(x)})

def distance(x, y):
	help = 0
	if len(x) != len(y):
		return -1
	for i in range(len(x)):
		help += (y[i]-x[i])**2
	return math.sqrt(help)

def distanceToDataset(x, data):
	result = []
	for datapoint in data:
		result.append(distance(x, datapoint))
	return result

def knn(k, x, data, datalabels):
	distances = np.array(distanceToDataset(x, data))
	indices = distances.argsort()[:k]
	nn = []
	for i in indices:
		nn.append(datalabels[i])
	try:
		return mode(nn)
	except StatisticsError:
		return datalabels[distances.argsort()[:1][0]]

def knn_list(k, xs, data, datalabels):
	result = []
	for x in xs:
		result.append(knn(k, x, data, datalabels))
	return result

def how_good_is_k(k, xs, xslabels, data, datalabels):
	guesses = knn_list(k, xs, data, datalabels)
	n = 0
	for i in range(len(guesses)):
		if guesses[i] != xslabels[i]:
			n += 1
	return n/len(guesses)*100

def get_the_best_k(xs, xslabels, data, datalabels):
	for x in range(len(datalabels)):
		print(x, how_good_is_k(x, xs, xslabels, data, datalabels))

#get_the_best_k(valdata, vallabels, data, labels)

print(knn_list(58, daysdata, data, labels))