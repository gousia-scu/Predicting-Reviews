
import numpy as np
from math import exp
X_train=np.load('train.mat')
y=np.loadtxt('train.labels')
X_test=np.load('test.mat')

def cost_func(theta,y,X):
         z = y * X.dot(theta)
         return np.mean(np.log(1.0 + np.exp(-z)))

def sigmoid(x):
         return 1/(1+(np.exp(-x)))

def predict(X,w):
        y_predicted_cls=[]
        z=np.dot(X, w)
        y_test=sigmoid(z)
        for i in y_test:
            if i>=0.5:
                y_predicted_cls.append(+1)    
            elif i<0.5:
                y_predicted_cls.append(-1)
        return y_predicted_cls  
   
def fit(X,y):
        theta = np.zeros(X.shape[1])
        for _ in range(18000):
            z=np.dot(X, theta)
            y_predicted=sigmoid(z)
            C=cost_func(theta,y,X) #passing the arguments to calculate cost
            print('cost_function',C)
            g = np.dot(X.T,(y_predicted-y))/X.shape[0] #gradient function
            theta=theta-0.3*g # calculating gradient descent
        print("Final Cost : ",C)
        print("Final Parameters : {0},{1},{2}".format(theta[0],theta[1],theta[2]))
        return theta

#fitting the given training set in the built model
true=fit(X_train,y) 
#predicting the new data which is the test.mat file using the obtained theta values from fitting the model
predictions=predict(X_test,true)      
#creating a new text file for entering the values obatined from predicting the data of test file       
with open('final.txt', 'w') as f:
            f.write('\n'.join(str(element) for element in predictions))





