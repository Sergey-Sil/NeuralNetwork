


import numpy as np

def load_data_normalize():
	lines = [[int(i) for i in line.split(',')] for line in open("mnist_train.csv", 'r')]

	training_inputs = np.array([np.reshape(x[1:], (784, 1)) for x in lines])
	training_inputs = training_inputs.astype('float32')
	training_inputs /=255
	tr_tar = np.array([i[0] for i in lines])
	training_results = [vectorized_result(y) for y in tr_tar]
	training_data = zip(training_inputs, training_results)

	lines = [[int(i) for i in line.split(',')] for line in open("mnist_test.csv", 'r')]

	test_inputs = np.array([np.reshape(x[1:], (784, 1)) for x in lines])
	test_inputs = test_inputs.astype('float32')
	test_inputs /=255
	test_target = np.array([i[0] for i in lines])
	test_data = zip(test_inputs, test_target)

	training = [i for i in training_data]
	test = [i for i in test_data]

	return training, test
	
def load_data():
	lines = [[int(i) for i in line.split(',')] for line in open("mnist_train.csv", 'r')]

	training_inputs = np.array([np.array(x[1:]) for x in lines])
	training_target = np.array([i[0] for i in lines])

	lines = [[int(i) for i in line.split(',')] for line in open("mnist_test.csv", 'r')]

	test_inputs = np.array([np.array(x[1:]) for x in lines])
	test_target = np.array([i[0] for i in lines])
	return training_inputs, training_target, test_inputs, test_target

def vectorized_result(j):
	e = np.zeros((10, 1))
	e[int(j)] = 1.0
	return e
 
