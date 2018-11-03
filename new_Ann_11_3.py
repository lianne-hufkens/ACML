import numpy as np

m1 = 8 #+1 bias
m2 = 3 #+1 bias
m3 = 8

w1 = np.random.rand(m1+1,m2) #9X3
w2 = np.random.rand(m2+1,m3) #4X8

alpha = .75
lambdaDecay = 0

def sigmoid(z):
    return 1/(1+np.exp(-z))


def calculateDeltaSimple(out, expect):
    return out - expect


def derive(a):
    return (a * (1-a))



in1 = np.array(([1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1])) #8x1

out3=in1            #our expected output same as the input

print("Initial Weight:")
print("W1 for first layer and second layer")
print(w1)
print("W2 for second layer and 3rd layer")
print(w2)

z2 = np.dot(np.c_[np.ones(8),in1], w1)          #add bias
a2 = sigmoid(z2)
z3 = np.dot(np.c_[np.ones(8), a2], w2)          #add bias
a3 = sigmoid(z3)

print("Original_Output:")
print(a3)


for i in range(100000):
    z2 = np.dot(np.c_[np.ones(8), in1], w1)     #add bias
    a2 = sigmoid(z2)
    z3 = np.dot(np.c_[np.ones(8), a2], w2)      #add bias
    a3 = sigmoid(z3)
    derivation_g3=derive(a3)
    delta3=calculateDeltaSimple(a3,out3)
    der_cost3=np.multiply(delta3,derivation_g3)
    modify_a2=np.c_[np.ones(8),a2]      #add bias
    new_min_cost23=np.dot(modify_a2.T,der_cost3)

    derivation_g2 = derive(a2)
    modify_w2=w2[1:]                #without bias
    der_cost2=np.dot(der_cost3,modify_w2.T*derivation_g2)     #only change here multiply with the transpose of all weight w2 without bias
    modify_input = np.c_[np.ones(8), in1] #add bias
    new_min_cost12=np.dot(modify_input.T,der_cost2)

    w1=w1-alpha*new_min_cost12+lambdaDecay  #i didn't use 1/m as it's also a constant
    w2=w2-alpha*new_min_cost23+lambdaDecay # same for lamda



print("Complete the traning:")
print("W1 for first layer and second layer")
print(w1)
print("W2 for second layer and 3rd layer")
print(w2)

z2 = np.dot(np.c_[np.ones(8),in1], w1)      #add bias
a2 = sigmoid(z2)
z3 = np.dot(np.c_[np.ones(8), a2], w2)      #add bias
a3 = sigmoid(z3)

print("Original_Output:")
print(a3)
