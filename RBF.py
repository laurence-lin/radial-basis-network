import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
import pandas as pd
import random as rd
plt.close('all')

'''IBM stock data'''
IBM = pd.read_csv('IBM_stock_close_price.csv')
ibm = np.array(IBM)
plt.figure(7)
plt.plot(ibm[:,1])
plt.title('IBM close price 1961~1962')
plt.xlabel('days')
plt.ylabel('close price')

Max = max(IBM.iloc[:,1])
Min = min(IBM.iloc[:,1])

for i in range(ibm.shape[0]):
    ibm[i,1] = (ibm[i,1] - Min) / (Max - Min)

ibm_train = ibm[0:250,1]
ibm_test = ibm[250:,1]


'''initialized center'''
Xmax = 1
Xmin = 0
hidden_neuron = 15; '''hidden neuron for encoding to a single input variable'''
center = []
for neuron in range(hidden_neuron):
    center.append( Xmin + (Xmax - Xmin)/(2*hidden_neuron) + (neuron - 1)*((Xmax - Xmin)/hidden_neuron) )
    center[neuron] = float('%.4f' % center[neuron]) # get float number by 4 decimal point
    '''10 RBF centers'''

print('Initial center =', center)
#dmax = (2*(hidden_neuron-1) - 1)/(2*hidden_neuron)
dmax = max(center) - min(center)
width = dmax/(2*hidden_neuron)**0.5
Width = width*np.ones((1,hidden_neuron)) # width array for optimization
beta = - 1 / (2 * (width ** 2))
delta = width*np.ones((hidden_neuron,1))

class RBFNN:

    def __init__(self, n_input, n_hidden, w_in_hidden = None, w_out = None):
        self.n_input = n_input  # practically, input neurons depend on sample features. Set up first is unnecessary
        self.n_hidden = n_hidden
        self.w_in_hidden = w_in_hidden  # usually set = 1
        '''3 parameters to be optimized during iteration, thus give them self attribute'''
        self.w_out = w_out  # randomly initialized weight
        self.center = center
        self.delta = delta
        self.output = []
        self.hidden_out = []

    def gaussian(self, input, Center):  # calculate 1 * 5 array input mapping to 10 * 5 output matrix
        '''input x dimension = number of features'''
        encode_neuron = self.n_hidden
        '''I don't know how to set the width theoretically'''
        hidden_out = np.zeros((encode_neuron, 1))  # 10*1 matrix
        for j in range(encode_neuron):
            hidden_out[j, 0] = np.exp(-self.Euclidean(input, Center[j]) ** 2 / 2 * (self.delta[j] ** 2))

        return hidden_out  # 10 outputs of hidden neuron

    def Euclidean(self, a, b):
        return norm(a - b)  # return euclidean distance of 2 vector

    def HiddenLayer(self, input):
        centroid = self.center
        hidden_output = self.gaussian(input,centroid)  # hidden neurons output of 1 input sample
        self.hidden_out = hidden_output
        return hidden_output # 10*1 matrix

    def feed_forward(self, input):  # feed forward calculation from input to output layer, calculate one sample at a time
        hidden_out = self.HiddenLayer(input) # 10*1 array
        weight = self.w_out # 1*10 array
        output = np.dot(weight, hidden_out)
        return output # output value of single sample

    def train(self, train_sample, sample_target):

        # Gradient descent learning: weight, center, width
        # update self attribute, keep the update to next iteration
        # Weight update:
        lr = 0.2  # learning rate
        num_hidden = self.n_hidden

        samples = train_sample.shape[0]
        w_gradient = np.zeros((num_hidden, 1))
        c_gradient = np.zeros((num_hidden, 1))
        delta_gradient = np.zeros((num_hidden, 1))
        beta = np.zeros((num_hidden,1))
        Error = 0 # total error

        for sample in range(samples):
            output = self.feed_forward(train_sample[sample,:]) # calculate one output at a time
            Error += output - sample_target[sample]
            # Batch learning

            for j in range(num_hidden):
                beta[j] = ( -1 / (2*(self.delta[j]**2)) )

                w_gradient[j] = w_gradient[j] + (output - sample_target[sample])*\
                                                np.exp( beta[j] * (self.Euclidean(train_sample[sample,:], self.center[j]) ** 2) )
                c_gradient[j] = c_gradient[j] + (output - sample_target[sample])*self.w_out[0,j]*beta[j]*( self.Euclidean(train_sample[sample,:],self.center[j] )) \
                                  *np.exp( beta[j] * (self.Euclidean(train_sample[sample,:], self.center[j]) ** 2) )
                delta_gradient[j] = delta_gradient[j] + (output - sample_target[sample])*(self.Euclidean(train_sample[sample,:],self.center[j])**2)\
                                                        *self.w_out[0,j]*( (self.delta[j])**(-3) )* \
                                                        np.exp(beta[j] * (
                                                        self.Euclidean(train_sample[sample, :], self.center[j]) ** 2))
                if self.delta[j] > 0:
                   if delta_gradient[j] < 0:
                      print('error =',output - sample_target[sample])

        for i in range(num_hidden):
           self.w_out[0,i] += -lr*w_gradient[i]/samples
           self.center[i] += 2*lr*c_gradient[i]/samples
           self.delta[i] += -lr*delta_gradient[i]/samples


        #print('w_out =', self.w_out)
        #print('center =', self.center)
        print('delta =', self.delta)


        return Error # scale in [0,1]

    def predict(self, test_data):
        # predict for one test data
        estimate = self.feed_forward(test_data)
        # decode
        estimate = Min + estimate * (Max - Min)
        return estimate

if __name__ == '__main__':

   Max = max(IBM.iloc[:,1])
   Min = min(IBM.iloc[:,1])
   # network parameters
   # Use close price as prediction
   slide_window = 3  # days of data to predict one future value, which is seen as feature
   duration = ibm_train.shape[0] - slide_window # days duration of whole predict process
   #duration = 1
   input_neurons = slide_window # number of features = input neurons number
   hidden_neurons = 15
   weight = np.zeros((1,hidden_neurons)) # weight = 1*10 matrix
   for j in range(hidden_neurons):
       weight[0,j] = rd.uniform(0,1)

   rbf = RBFNN(input_neurons, hidden_neurons, None, weight) # build an RBF network archictecture

   # create training set time series
   samples = duration
   train_set = np.zeros((duration,slide_window))
   target = np.zeros((samples,1))

   for sample in range(samples):
       train_set[sample,:] = ibm_train[sample:(sample + slide_window)]
       target[sample] = ibm[sample + slide_window,1]


   Iteration = 100
   Error = 0
   Performance = np.zeros((Iteration,1))
   Predict = np.zeros((Iteration,1))
   for iterate in range(Iteration):
       error = 0
       # Batch learning
       error = rbf.train(train_set, target)  # output = total error
       E = error * (Max - Min) / samples + Min
       print('Error in epoch: ', E)
       Performance[iterate] = E
       Predict[iterate] = rbf.predict(train_set[0, :])

   #print(Performance)
   plt.figure()
   plt.title('Error Performance')
   plt.plot(Performance)

   # training set performance
   check = np.zeros((train_set.shape[0], 1))
   for i in range(train_set.shape[0]):
       check[i] = rbf.predict(train_set[i,:])
       target[i] = Min + (Max - Min)*target[i]

   # test set performance
   test_duration = ibm_test.shape[0] - slide_window
   test_set = np.zeros((test_duration,slide_window))
   for sample in range(test_set.shape[0]):
       test_set[sample,:] = ibm_test[sample:sample + slide_window]
   test_target = ibm_test[slide_window:]
   test_out = np.zeros((train_set.shape[0], 1))
   for i in range(test_set.shape[0]):
       test_out[i] = rbf.predict(test_set[i, :])
       test_target[i] = Min + (Max - Min) * test_target[i]

   plt.figure()
   plt.title('test performance')
   plt.plot(test_target, 'r-', label='Real output')
   plt.plot(test_out, 'b-', label='Predict output')
   plt.legend(loc='upper right')

   plt.show()