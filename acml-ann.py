import numpy as np
from scipy.special import expit #aka sigmoid function

m1 = 8 #+1 bias
m2 = 3 #+1 bias
m3 = 8

b1 = np.random.rand(m2, 1) #3x1
w1 = np.random.rand(m2, m1) #3x8

b2 = np.random.rand(m3, 1) #8x1
w2 = np.random.rand(m3, m2) #8x3

alpha = 0.8
lambdaDecay = 0.1

def generateInputMatrix(num):
    base = np.array([1,0,0,0,0,0,0,0])
    return (np.array([np.roll(base, i).T for i in range(num)])).T

def calculateDeltaOut(y, a):
    return (a-y)

def calculateDelta(w, delta, z):
    #with arrays, use dot for matrix and multiply for element-wise multiplication
    return np.array((np.multiply(np.dot(w.T,delta),derive(z))), dtype=np.float64)

def derive(a):
    a = np.array(a, dtype=np.longdouble)
    return np.multiply((1-a), a)

def updateBias(b, dB):
    global alpha, m
    return b - alpha*(dB/m)

def updateWeight(w, dW):
    global alpha, m, lambdaDecay
    return w - alpha*((dW/m) + (lambdaDecay * w))

def printMatrices():
    global b1, w1, b2, w2
    print("Weights:")
    print(w1)
    print(w2)
    

#number of samples: change as necessary
m = input("Enter the number of samples to use (minimum is 4): ")
if m == "" or int(m) < 4:
    print("Input is invalid. Using m = 4...")
    m = 4
m = int(m)

alpha = input("Set Alpha (default 0.8): ")
if alpha == "" or int(alpha) <= 0 or int(alpha) >= 1:
    print("Using default Alpha = 0.8")
    alpha = 0.8
alpha = int(alpha)

lambdaDecay = input("Set Lambda (default 0.1): ")
if lambdaDecay == "" or int(lambdaDecay) <= 0 or int(lambdaDecay) >= 1:
    print("Using default Lambda = 0.1")
    lambdaDecay = 0.1
lambdaDecay = int(lambdaDecay)

xAll = generateInputMatrix(m) #8xm
    
run = True
loopCount = 0

printMatrices()

while run:
    #set big deltas to zero
    dB1 = np.zeros((3,1)) #3x1
    dW1 = np.zeros((3,8)) #3x8
    dB2 = np.zeros((8,1)) #8x1
    dW2 = np.zeros((8,3)) #8x3
        
    yAll = np.array([])
    #for each of the training examples
    for x in xAll.T:
        x = np.array([x]).T
        a1 = x
        y = x #output needs to be the same as input
        
        #calculate output values
        z2 = np.dot(w1,a1)+b1 #3x1
        a2 = expit(z2) #3x1

        z3 = np.dot(w2,a2)+b2 #8x1
        a3 = expit(z3) #8x1

        if yAll.size == 0:
            yAll = a3
        else:
            yAll = np.hstack((yAll, a3))

        ##start backpropagation
        #calculate small deltas
        delta3 = calculateDeltaOut(y, a3) #8x1
        delta2 = calculateDelta(w2, delta3, z2) #3x1

        #partial derivatives
        pDB2 = delta3 #8x1
        pDW2 = np.dot(delta3, a2.T) #8x3
        pDB1 = delta2 #3x1
        pDW1 = np.dot(delta2, a1.T) #3x8

        #update big deltas
        dB2 = dB2 + pDB2 #8x1
        dW2 = dW2 + pDW2 #8x3
        dB1 = dB1 + pDB1 #3x1
        dW1 = dW1 + pDW1 #3x8

    #update all weights simultaneously
    b2 = updateBias(b2, dB2) #8x1
    w2 = updateWeight(w2, dW2) #8x3
    b1 = updateBias(b1, dB1) #3x1
    w1 = updateWeight(w1, dW1) #3x8

    loopCount += 1

    #check the difference between actual and expected output
    #if all entries are zeros or limit reached: stop loop
    if not delta3.any() or loopCount == 100000: 
       run = False 
       break

    if(loopCount % 10000 == 0):
        print("At loop "+str(loopCount)+" the error rate is "+str(np.mean(np.abs(delta3))))
        print("Input matrix: ")
        print(xAll)
        print("Output matrix: ")
        print(yAll)
        printMatrices()


        
print("Reached "+str(loopCount)+" iterations.")
