import numpy as np

layer1 = 8
layer2 = 3
layer3 = 8

w12 = np.random.rand(layer2, layer1+1) #3x9
w23 = np.random.rand(layer3, layer2+1) #8x4



def sigmoid(z):
    return 1/(1+np.exp(-z))
sigmoid_v = np.vectorize(sigmoid)

def addBias(matrix):
    return np.insert(matrix, 0, 1)

def generateAnswer(q):
    return np.roll(q, 1)

def calculateDeltaSimple(out, expect):
    return out - expect

def calculateDelta(weights, delta, outputs):
    weights = np.transpose(weights)
    return np.dot(np.dot(weights, delta), outputs) #with arrays, use dot for matrix mult.

def derive(a):
    return (a * (1-a))
derive_v = np.vectorize(derive)




inputs = np.array([1,0,0,0,0,0,0,0])[np.newaxis]
outputs1 = sigmoid_v(np.dot(w12, addBias(inputs))) #matrix*vector
outputs2 = sigmoid_v(np.dot(w23, addBias(outputs1))) #matrix*vector
delta3 = calculateDeltaSimple(outputs2, generateAnswer(outputs2))
delta2 = calculateDelta(w23, delta3, derive_v(addBias(outputs1)))
print(delta2)

