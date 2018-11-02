import numpy as np

m1 = 8 #+1 bias
m2 = 3 #+1 bias
m3 = 8

w1 = np.random.rand(m2, m1+1) #3x9
w2 = np.random.rand(m3, m2+1) #8x4
dB1 = np.zeros((3,1)) #3x1
dW1 = np.zeros((3,8)) #3x8
dB2 = np.zeros((8,1)) #8x1
dW2 = np.zeros((8,3)) #8x3

alpha = 0.5
lambdaDecay = 0.1

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
    #with arrays, use dot for matrix multiplication
    return np.multiply(np.dot(weights, delta), outputs)

def derive(a):
    return (a * (1-a))
derive_v = np.vectorize(derive)

def updateBias(b, m, dB):
    global alpha
    return b - alpha*((1/m) * np.mat(dB).T)

def updateWeight(w, m, dW):
    global alpha, lambdaDecay
    return w - alpha*(((1/m) * dW) + (lambdaDecay * w))


in1 = np.array([1,0,0,0,0,0,0,0])[np.newaxis] #8x1
out1 = addBias(in1) #9x1

in2 = np.dot(w1, out1) #3x1
out2 = sigmoid_v(in2) #4x1
a2 = addBias(out2) #4x1

in3 = np.dot(w2, a2) #8x1
out3 = sigmoid_v(in3) #8x1

##start backpropagation

#calculate small deltas 
delta3 = calculateDeltaSimple(out3, generateAnswer(out3)) #8x1
delta2 = calculateDelta(w2, delta3, derive_v(addBias(in2))) #4x1

#partial derivatives
pDB2 = delta3 #8x1
pDW2 = np.transpose(np.mat(delta3)) * np.mat(out2) #8x3

#update big deltas
dB2 = np.mat(dB2).T + pDB2 #8x1
dW2 = dW2 + pDW2 #8x3

#update the weights for layer 2->3
w2 = np.concatenate((updateBias(w2[:,:1], 1, dB2), updateWeight(w2[:,1:], m2, dW2)), axis=1)

###################################################
### code after this is still under construction ###
###################################################

#3x1
#3x8


pDB1 = delta2 #4x1
pDW1 = np.mat(in1).T * np.mat(delta2) #8x4

print(pDB1.shape)
print(pDW1.shape)

#print(w1.shape)
dB1 = np.mat(dB1).T + pDB1 
dW1 = dW1 + pDW1 

w1 = np.concatenate((updateBias(w1[:,:1], 1, dB1), updateWeight(w1[:,1:], m1, dW1)), axis=1)



