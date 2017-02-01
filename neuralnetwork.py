from mnist import MNIST
from sklearn import preprocessing
import numpy as np
import time
import matplotlib.pyplot as plt
# What follows is a generic neural network implementation, with learning
# rate decay and bias terms, using parameters from the 
# MNIST data set for didactic purposes. You can use
# any dataset, just interchanging the parameters below.
#
# This implementation will have one hidden layer, however adding
# more follows simiarly.
# Michael McPhillips, mcphillips@berkeley.edu
"""
obs is number of observations
dim is number of features

NUM_CLASSES is the number of classes
inp is the number of input variables
hid is the number of nodes in the hidden layer
out is the number of nodes in the output layer
"""
obs = 60000
dim = 10000

NUM_CLASSES = 10
inp = 784
hid = 200
out = 10
"""
PART I
i. First we will load the data set, 
ii. Append our bias vectors to the training and test set, 
iii. One-hot encode our labels,
iv. Initialize V and W through normal distributions,
		here I mean V and W to be the weight arrays found
		from the typical backpropogation proofs.
		V has dimension Nhid by Ninp + 1 randoms
		W has dimension Nout by Nhid + 1 randoms
"""
def loadDataset():
	""" Loading data from the MNIST dataset and returning
	 the testing and training datasets,
	and the testing and training labels"""
	mndata = MNIST('./data/')
	X_train, labels_train = map(np.array, mndata.load_training())
	X_test, labels_test = map(np.array, mndata.load_testing())
	return X_train, labels_train, X_test, labels_test

def initializeVW():
	""" Recal that our V will be HID by INP + 1 in dimensions and
	W will be OUT by HID + 1 in dimensions. We will initialize using a 
	normal disribution, with the following parameters."""
	return np.random.normal(0, 0.1, (hid, inp+1)), np.random.normal(0, 0.1, (out, hid+1))

def addBias(Xtrain, Xtest):
	""" This will add a bias term to XTRAIN and XTEST. It is a column
	of all ones. Make sure later on that they do not get any edge weights
	feeding into them. Then they will turn from a bias term into normal weights
	that are simply initialized to one."""
	a = np.ones(obs)
	a.shape = (obs, 1)
	a1 = np.ones(dim)
	a1.shape = (dim, 1)
	Xtrain = np.concatenate((Xtrain, a), axis=1).T
	Xtest = np.concatenate((Xtest, a1), axis= 1).T
	return Xtrain, Xtest

def oneHot(labels_train):    
	""" ONE_HOT will take in a vector of training labels, LABELS_TRAIN, 
	and will output an array of one_hot encoded training labels. """  
	return np.eye(NUM_CLASSES)[labels_train]
"""
PART II
This will include the first part of the algorithm:
i. The activation of the hidden layer.
ii. The activation of the output layer.
iii. A forward pass of the neural network.
"""


def hidden_act(x, V):
	""" The activation of the hidden layer, which takes
	the dot product of the inputs and the hidden layer,
	which is 200 by 785 multiplied by 785 by 1."""
	return relu(np.dot(V, x))

def output_act(h, W):
	""" The activation of the output layer: first we take
	the dot product of the output weight wo and h,
	which is 10 by 201 times 201 by 60,000. Then we apply
	softmax activation to it. This is commonly used in classification
	problems of this sort since the outputs will sum to one. 

	We appended the bias to the hidden layer here.
	NOTE: We make sure to append the bias at this step, or else
	our dot product will cause an edge to feed into the bias. This will
	turn it into a normal weight intialized at one.
 	The result is 10 by 60,000."""
	a = np.ones(h.shape[1])
	a.shape = (1, h.shape[1])
	h = np.concatenate((h, a), axis=0)
	return softmax1(np.dot(W,h))

def neuralnet(x, V, W):
	""" This is a forward pass of the neural network, which finds the
	sums of inputs starting hidden weights, activates, then finds sums 
	of the next layer and activates with softmax."""
	return output_act(hidden_act(x,V),W)

"""
PART III
This will include the second part of the algorithm:
i. Defining the particular function.
ii. Defining the predict function.
iii. A function to get the Loss of our current V and W.
iv. A helper function to get the inner partial derivatives.

"""

def cost(y,t):
	"""note that np.inner(a,b) = sum(a[:]*b[:]),
	 calculates cost of y and t where y is our true labels,
	  t is what our neural net produced."""
	return -np.sum(y*np.log(t))

def nn_predict(x, W, V):
	"""A predict function, using input X, our W, and our V,
	which have been produced through training. This just predicts
	the correct class as the one with the highest value out of the
	softmax layer."""
	return np.argmax(neuralnet(x,V,W), axis = 0)

def getloss(X, Y, W, V):
	"""GETLOSS will return the loss of our current V and W."""
	c = 0
	ans = neuralnet(X, V, W)
	for i in range(X.shape[1]):
		c = c + cost(Y[i], ans[:,i])
	return c

def helper(a, b):
	"""HELPER is used in getting the inner dels, or partial
	derivatives of the output layer."""
	for i in range(0, a.shape[1]):
		a[:,i] = a[:,i]*b[i]
	return a

"""
PART IV
i. Defines the training function which will update paramters,
run predict every 1000 iterations, recording the loss and accuracy.
ii. Updates the W and V by gradient descent, using RATE as the learning
rate in the gradient descent algorithm
iii. Returns the accuracy of our current neural network, running predict on
the current weights, and carrying in the correct Y labels.
iv. Test for accuracy with a simple loop of the preceding values, checking to
see how many are equivelant and summing.
"""
def train(X, Y, ylabels, W, V, rate, iterations):
	"""TRAINS the dataset using the specified neural network, takes in
	X as the training dataset, Y as the answers, ylabels as the numeric labels,
	W and V as the weight arrays, and RATE as the learning rate. It will return
	W and V, the correct weight arrays, X the number of iterations ran,
	Y the loss at that iteration, and Z the accuracy at that iteration.

	This version employs a learning rate decay, observe integer J."""
	x = []
	y = []
	z = []
	J = 1
	for k in range(0, iterations):
		i = np.random.random_integers(0,X.shape[1]-1)
		if k % 50000 == 0:
			J = J + 1
		W, V = update(X_train[:,i], ytrain[i], W, V, rate/J)
		if k % 1000 == 0:
			x.append([k])
			y.append([getloss(X, Y, W, V)])
			z.append([getacc(X, ylabels, W, V)])
	return W, V, x, y, z

def update(X_train, ytrain, W, V, rate):
	"""UPDATES the given W and V weight vectors through running one
	iteration of gradient descent on the X_TRAIN data, using YTRAIN as
	the correct labels, at learning rate RATE. This will return the
	new W array WNEW and the new V array VNEW. """
	zh = np.dot(V, X_train)
	h = relu(zh)
	a = np.ones(1)
	hhat = np.concatenate((h, a), axis=0)
	y = np.dot(W, hhat)
	#y = output_act1(h, W)
	#dels are 10x1
	delout = softmax(y) - ytrain
	#gradW is 10x1 by 201 x 1
	GradW = np.outer(delout, hhat)
	#delin is 200x1, need to remove bias from GradW
	#so it is 200x10 times 10x1 = 200 x 1***
	hprime = reluprime(zh)
	##hprime is g'(s^(l-1))
	Wnew = W - rate*GradW
	WforV = helper(np.delete(W, 200, 1), hprime)
	delin = np.dot(WforV.T, delout)
	# 200 x 1 * 785 * 1 = 200 x 785
	GradV = np.outer(delin, X_train)
	Vnew = V - rate*GradV
	return Wnew, Vnew


def getacc(X, Y, W, V):
	"""Simply runs predict on the current data X, and weight
	vectors W and V to get the predicted answers. Also
	carries in the correct Y. Feeds both into test."""
	given = nn_predict(X, W, V)
	expected = Y
	return test(given, expected)

def test(given, expected):
	""" Calculates how many entries in GIVEN are different
	from EXPECTED. """
	c = 0
	for i in range(0,given.shape[0]):
		if given[i] == expected[i]:
			c = c + 1
	return c/given.shape[0]

"""
Here are some basic activation functions that are commonly used. You don't
need to use these, but they do need to be nonlinear functions. Otherwise your
post-activations are simply linear combinations of pre-activations. Softmax for
the output layer is good since it yields a result which sums to one, which can
be interpereted as a confidence estimate.
"""
def relu(z):
	"""The RELU function takes in a vector Z and returns the image after
	applying the relu function, which is rectified linear units."""
	return np.maximum(z, 0)

def reluprime(h):
	"""The derivative of the RELU function, needed for backpropogation. We
	used relu on H, the hidden layer."""
	z = [0]*len(h)
	for i in range(0, len(h)):
		if h[i] > 0:
			z[i] = 1
		else:
			z[i] = 0
	return np.array(z)

def softmax(z):
	"""The SOFTMAX function takes in a vector Z and returns the image after
	applying the SOFTMAX function. """
	curr = np.argmax(z)
	maxim = z[curr]
	return np.exp(z - maxim)/np.sum(np.exp(z - maxim), axis = 0)


if __name__ == "__main__":

	""" Runs all of the above, loading training and testing
	the neural network and printing accruacies and time. Also
	plotting iterations vs. accuracy and iterations vs. loss.
	"""
	time1 = time.time()
	X_train, labels_train, X_test, labels_test = loadDataset()
	V, W = createVW()
	ytest = oneHot(labels_test)
	ytrain = oneHot(labels_train)
	RATE = 0.1
	iterations = 10000
	W, V, x, y, z = train(X_train, ytrain, labels_train, W, V, RATE, iterations)
	prediction = nn_predict(X_train, W, V)
	print(test(ans, labels_train))
	print(RATE)
	ans1 = nn_predict(X_test, W, V)
	print(test(ans1, labels_test))
	time2 = time.time()
	print(time2 - time1)
	plt.plot(x, y)
	plt.show()
	print(x)
	print(y)
	print(z)
	#Last two lines to save your weights as a CSV
	#np.savetxt("W.csv", W, delimiter=",")
	#np.savetxt("W.csv", W, delimiter=",")
