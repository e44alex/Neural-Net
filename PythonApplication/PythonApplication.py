import numpy as np
import numpy.random as rand

def sigmoid(x):
    return 1/(1+np.exp(-x))

input = np.array([1,0]).T
weights1 = np.array([[0.45,0.78],
                     [0.12,0.13]])
weights2 = np.array([1.5,-2.3])
outputs = np.array([])
ideal_output = 1

for iter in range(10000):

    H1input = (input[0] * weights1[0][0]) + (input[1] * weights1[1][0])
    H2input = (input[0] * weights1[0][1]) + (input[1] * weights1[1][1])

    H1output = sigmoid(H1input)
    H2output = sigmoid(H2input)
    
    O_input = (H1output * weights2[0]) + (H1output * weights2[1])
    O_output = sigmoid(O_input)
    
    outputs = np.append(outputs,O_output)
    if O_output != ideal_output:
        weights1 = rand.sample((2,2))
        weights2 = rand.sample(2) * 10


MSEn = 0
n=0
MSE = 0
for output in outputs:
    MSEn += pow((ideal_output-output),2)
    n+=1

if n!= 0:
    MSE = MSEn / n

print("Error percent: " + str(MSE*100))
a = outputs.max()
print(a)