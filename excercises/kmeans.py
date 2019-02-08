import numpy as np
import math
import random
import matplotlib.pyplot as plt

data = np.genfromtxt('../resources/dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5:lambda x: 0 if x == b'-1' else float(x), 7:lambda x: 0 if x == '-1' else float(x)})
data = data.tolist()

dates = np.genfromtxt('../resources/dataset1.csv', delimiter=';', usecols=[0])
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

labeleddata = []
for i in range(len(data)):
	h = data[i]+[labels[i]]
	labeleddata.append(h)

valdata = np.genfromtxt('../resources/validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5:lambda x: 0 if x == b'-1' else float(x), 7:lambda x: 0 if x == '-1' else float(x)})
valdata = valdata.tolist()

valdates = np.genfromtxt('../resources/validation1.csv', delimiter=';', usecols=[0])
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

labeledvaldata = []
for i in range(len(valdata)):
	h = valdata[i]+[vallabels[i]]
	labeledvaldata.append(h)

def distance(x, y):
	help = 0
	if len(x) != len(y):
		return -1
	for i in range(len(x)):
		help += (y[i]-x[i])**2
	return math.sqrt(help)

def mean(xs):
	result = []
	for i in range(len(xs[0])):
		helper = 0
		for x in xs:
			helper += x[i]
		result.append(helper/len(xs))
	return result

def randomCentroids(k, dataset):
	npdataset = np.transpose(np.array(dataset))
	maxval = [max(val) for val in npdataset]
	minval = [min(val) for val in npdataset]
	result = [[random.randint(minval[i], maxval[i]) for i in range(len(maxval))] for n in range(k)]
	return result

def assignCentroidsLabeledData(dataset, centroids):
	result = [[centroid, []] for centroid in centroids]
	for point in dataset:
		distanceToCentroids = [distance(point[:-1], centroid) for centroid in centroids]
		indexClosestCentroid = np.argmin(np.array(distanceToCentroids))
		result[indexClosestCentroid][1].append(point)
	return result

def assignCentroids(dataset, centroids):
	result = [[centroid, []] for centroid in centroids]
	for point in dataset:
		distanceToCentroids = [distance(point, centroid) for centroid in centroids]
		indexClosestCentroid = np.argmin(np.array(distanceToCentroids))
		result[indexClosestCentroid][1].append(point)
	return result

def getNewCentroidsLabeledData(centroidData):
	for cluster in centroidData:
		for point in cluster[1]:
			point.pop()
	result = [mean(cluster[1]) for cluster in centroidData]
	return result

def getNewCentroids(centroidData):
	result = [mean(cluster[1]) if cluster[1] else cluster[0] for cluster in centroidData]
	return result

def doTheThingNTimes(dataset, labeledDataset, k, n):
	centroids = randomCentroids(k, dataset)
	centroidData = None
	for p in range(n):
		centroidData = assignCentroids(dataset, centroids)
		centroids = getNewCentroids(centroidData)
	result = assignCentroidsLabeledData(labeleddata, centroids)
	return result

def aggregateIntraClusterDistance(centroidLabeledData):
	result = 0
	for cluster in centroidLabeledData:
		if cluster[1]:
			for point in cluster[1]:
				result += distance(point[:-1], cluster[0])**2
	return result

def printScreePlotData(dataset, labeledDataset, maxK, n):
	for k in range(maxK):
		print("k = %d: %.3f" % (k+1, aggregateIntraClusterDistance(doTheThingNTimes(dataset, labeledDataset, k+1, n))))

def getScreePlotData(dataset, labeledDataset, maxK, n):
	return [aggregateIntraClusterDistance(doTheThingNTimes(dataset, labeledDataset, k+1, n)) for k in range(maxK)]

def drawScreePlot(dataset, labeledDataset, maxK, n):
	plt.plot([i+1 for i in range(maxK)], getScreePlotData(dataset, labeledDataset, maxK, n), 'ro')
	plt.show()


# printScreePlotData(data, labeleddata, 50, 100)
# getScreePlotData(data, labeleddata, 100, 100)

drawScreePlot(data, labeleddata, 50, 100)

# Using the drawScreePlot function, you can see the correct amount of seasons would be around k=6, where the slope of the graph seems to be the closest to the "edge"