# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:00:51 2019

@author: YourAverageSciencePal
"""
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
import pickle
'''
Depending on your choice of library you have to install that library using pip
'''


'''
Read chapter on neural network from the book. Most of the derivatives,formulas 
are already there.
Before starting this assignment. Familarize yourselves with np.dot(),
What is meant by a*b in 2 numpy arrays.
What is difference between np.matmul and a*b and np.dot.
Numpy already has vectorized functions for addition and subtraction and even for division
For transpose just do a.T where a is a numpy array 
Also search how to call a static method in a class.
If there is some error. You will get error in shapes dimensions not matched
because a*b !=b*a in matrices
'''

class NeuralNetwork():
    @staticmethod
    #note the self argument is missing i.e. why you have to search how to use static methods/functions
    def cross_entropy_loss(y_pred, y_true):
        '''implement cross_entropy loss error function here
        Hint: Numpy has a sum function already
        Numpy has also a log function
        Remember loss is a number so if y_pred and y_true are arrays you have to sum them in the end
        after calculating -[y_true*log(y_pred)]'''
        # loss_sum = np.sum(y_true * np.log(y_pred)) 
        return -(y_true * np.log(y_pred)).sum()

    @staticmethod
    def accuracy(y_pred, y_true):
        '''function to calculate accuracy of the two lists/arrays
        Accuracy = (number of same elements at same position in both arrays)/total length of any array
        Ex-> y_pred = np.array([1,2,3]) y_true=np.array([1,2,4]) Accuracy = 2/3*100 (2 Matches and 1 Mismatch)'''
        counter = 0
        for i in range(len(y_pred)):
            if (y_pred[i] == y_true[i]).any():
                counter = counter + 1
        return (counter / len(y_pred)) * 100
    
    @staticmethod
    def softmax(x):
        '''Implement the softmax function using numpy here
        Hint: Numpy sum has a parameter axis to sum across row or column. You have to use that
        Use keepdims=True for broadcasting
        You guys should have a pretty good idea what the size of returned value is.
        '''
        my_exp = np.exp(x)
        sumOfexp = np.sum(my_exp, axis=1, keepdims=True)
        return my_exp / sumOfexp
    
    @staticmethod
    def sigmoid(x):
        '''Implement the sigmoid function using numpy here
        Sigmoid function is 1/(1+e^(-x))
        Numpy even has a exp function search for it.Eh?
        '''
        return 1 / (1 + np.exp(-x))
    
    def __init__(self):
        '''Creates a Feed-Forward Neural Network.
        "nodes_per_layer" is a list containing number of nodes in each layer (including input layer)
        "num_layers" is the number of layers in your network 
        "input_shape" is the shape of the image you are feeding to the network
        "output_shape" is the number of probabilities you are expecting from your network'''

        # Input, Hidden, Output
        self.num_layers = 3 # includes input layer
        # Input = 784, Hidden = 30, Output = 10
        self.nodes_per_layer = [784, 30, 10]
        self.input_shape = self.nodes_per_layer[0]
        self.output_shape = self.nodes_per_layer[2]
        self.__init_weights(self.nodes_per_layer)

    def __init_weights(self, nodes_per_layer):
        '''Initializes all weights and biases between -1 and 1 using numpy'''
        self.weights_ = []
        self.biases_ = []
        for i,_ in enumerate(nodes_per_layer):
            if i == 0:
                # skip input layer, it does not have weights/bias
                continue
            
            weight_matrix = np.random.uniform(low=-1,high=1,size=(nodes_per_layer[i-1],nodes_per_layer[i]))
            self.weights_.append(weight_matrix)
            bias_vector = None
            self.biases_.append(bias_vector)
    
    def fit(self, Xs, Ys, epochs, lr):
        '''Trains the model on the given dataset for "epoch" number of itterations with step size="lr". 
        Returns list containing loss for each epoch.'''
        history = []

        print(f"Training model for {epochs} epochs...")        
        for e in (range(epochs)):
            for i in range(Xs.shape[0]):
                reshapedX, reshapedY = Xs[i].reshape((1,self.input_shape)), Ys[i].reshape((1,self.output_shape))

                activations = self.forward_pass(reshapedX)
                deltas = self.backward_pass(reshapedY, activations)

                layer_inputs = [reshapedX] + activations[:-1]
                self.weight_update(deltas, layer_inputs, lr)
            
            cross_loss, epoch_accuracy = self.evaluate(Xs, Ys)
            history.append(cross_loss)

            print(f"Epoch Number {e+1} -------> {epoch_accuracy * Xs.shape[0]}/{Xs.shape[0]} images correctly classified")
            print(f"Accuracy {epoch_accuracy} % ------------------- Error {100-epoch_accuracy} %")
        
        return history
        
    def forward_pass(self, input_data):
        '''Executes the feed forward algorithm.
        "input_data" is the input to the network in row-major form
        Returns "activations", which is a list of all layer outputs (excluding input layer of course)
        What is activation?
        In neural network you have inputs(x) and weights(w).
        What is first layer? It is your input right?
        A linear neuron is this: y = w.T*x+b =>T is the transpose operator 
        A sigmoid neuron activation is y = sigmoid(w1.T*x+b1) for 1st hidden layer 
        Now for the last hidden layer the activation y = sigmoid(w2.T*y+b2).
        '''
        activations = []

        hid_layer = self.sigmoid(input_data @ self.weights_[0])

        out_layer = self.sigmoid(hid_layer @ self.weights_[1])
        
        activations = [hid_layer, out_layer]
        return activations
    
    def backward_pass(self, targets, layer_activations):
        '''Executes the backpropogation algorithm.
        "targets" is the ground truth/labels
        "layer_activations" are the return value of the forward pass step
        Returns "deltas", which is a list containing weight update values for all layers (excluding the input layer of course)
        You need to work on the paper to develop a generalized formulae before implementing this.
        Chain rule and derivatives are the pre-requisite for this part.
        '''
        deltas = []

        diff = layer_activations[1] - targets
        # diff * derivative 2
        delta_output = diff * (layer_activations[1] * (1 - layer_activations[1])) 

        product = self.weights_[1] @ delta_output.T
        # product * derivative 1
        delta_hidden = (product.T) * (layer_activations[0] * (1 - layer_activations[0])) 
        
        deltas = [delta_hidden ,delta_output] 
        return deltas
            
    def weight_update(self, deltas, layer_inputs, lr):
        '''Executes the gradient descent algorithm.
        "deltas" is return value of the backward pass step
        "layer_inputs" is a list containing the inputs for all layers (including the input layer)
        "lr" is the learning rate
        You just have to implement the simple weight update equation. 
        
        '''
        # Layer_input ka tranpose leh rhay because uski shape 1 x 784 hai aur deltas ki 1 x 30
        # Final result should be 784 x 30 as our self.weights[0] ki shape bhi 784 x 30 hai
        self.weights_[0] -= lr * ((layer_inputs[0]).T @ deltas[0])       

        # Layer_input ka tranpose leh rhay because uski shape 1 x 30 hai aur deltas ki 1 x 10
        # Final result should be 30 x 10 as our self.weights[1] ki shape bhi 30 x 10 hai 
        self.weights_[1] -= lr * ((layer_inputs[1]).T @ deltas[1])
        
    def predict(self, Xs):
        '''Returns the model predictions (output of the last layer) for the given "Xs".'''
        predictions = []
        for i in range(Xs.shape[0]):
            reshapedX = Xs[i].reshape((1,self.input_shape))
            predicted_result = (self.forward_pass(reshapedX)[-1]).reshape((self.output_shape))
            predictions.append(predicted_result)
        predictions = np.array(predictions)
        return predictions
    
    def evaluate(self, Xs, Ys):
        '''Returns appropriate metrics for the task, calculated on the dataset passed to this method.'''
        pred = self.predict(Xs)
        acc = self.accuracy(pred.argmax(axis=1), Ys.argmax(axis=1)) 
        loss = self.cross_entropy_loss(pred, Ys) 
        return loss,acc

    def give_images(self,listDirImages):
        '''Returns the images and labels from the listDirImages list after reading
        Hint: Use os.listdir(),os.getcwd() functions to get list of all directories
        in the provided folder. Similarly os.getcwd() returns you the current working
        directory. 
        For image reading use any library of your choice. Commonly used are opencv,pillow but
        you have to install them using pip
        "images" is list of numpy array of images 
        labels is a list of labels you read 
        '''
        images = []
        labels = []
        
        return images,labels
    
    def generate_labels(self,labels):
        '''Returns your labels into one hot encoding array
        labels is a list of labels [0,1,2,3,4,1,3,3,4,1........]
        Ex-> If label is 1 then one hot encoding should be [0,1,0,0,0,0,0,0,0,0]
        Ex-> If label is 9 then one hot encoding shoudl be [0,0,0,0,0,0,0,0,0,1]
        Hint: Use sklearn one hot-encoder to convert your labels into one hot encoding array
        "onehotlabels" is a numpy array of labels. In the end just do np.array(onehotlabels).
        '''
        onehotlabels = []
        onehot_encoder = OneHotEncoder(sparse=False)
        newLabels = labels.reshape(len(labels), 1)
        output = onehot_encoder.fit_transform(newLabels)
        onehotlabels = output
        return onehotlabels

    def save_weights(self,fileName):
        '''save the weights of your neural network into a file
        Hint: Search python functions for file saving as a .txt'''
        with open(fileName, 'wb') as f:
            pickle.dump(self.weights_, f)
        
    def reassign_weights(self,fileName):
        '''assign the saved weights from the fileName to the network
        Hint: Search python functions for file reading
        '''
        with open(fileName, 'rb') as f:
            self.weights_ = pickle.load(f)

    def savePlot(self):
        '''function to plot the execution time versus learning rate plot
        You can edit the parameters pass to the savePlot function'''
        pass

#Generate list
def read_file(file_name):
    mylist = []
    file = open(file_name,'r')
    i = 0
    while(True):
        word = file.readline()
        if (word ==''):
            break
        mylist.append([])
        final_word = ""

        while True:
            if '[' in word:
                word = word[1:]
                
            if ']' in word:
                word = word[:-2]
                final_word = final_word + word.replace('\n'," ")
                break
                
            final_word = final_word + word.replace('\n'," ")
            
            word = file.readline()
        
        res = final_word.split()
        int_res = list(map(int,res))
        mylist[i] = int_res
        i = i+1
    file.close()
    return mylist

def readlabels(file_name):
    readfileobj = open(file_name, 'r')
    line = readfileobj.readline()
    labels = []
    while line != '':
        labels.append(int(line))
        line = readfileobj.readline()
    return labels	


def main():
    condition = sys.argv[1]

    if condition == 'train':
        file_name = sys.argv[2]
        label_name = sys.argv[3]
        learning_rate = float(sys.argv[4])
        starttime = time.time()

        trainval = read_file(file_name)
        trainlabels = readlabels(label_name)
        trainvalNp = np.array(trainval) / 255
        trainlabelsNp = np.array(trainlabels)
        print ("Time taken to Read the Files is = ", str(time.time() - starttime) + " seconds")
        print()

        nn = NeuralNetwork()

        trainlabelsHot = np.array(nn.generate_labels(trainlabelsNp))
        starttime = time.time()    
    
        loss = nn.fit(trainvalNp, trainlabelsHot, 2, learning_rate)    
        print()
        print ("Execution Time is = ", str(time.time() - starttime) + " seconds")
        print()
        print("Data has been Trained")
        print()
        nn.save_weights("netWeights.txt")
        print("Weights have been Saved")
    
    elif condition == 'test':

        testfile = sys.argv[2]
        testlabels = sys.argv[3]
        netWeights = sys.argv[4]
        testval = read_file(testfile)
        testlabels = readlabels(testlabels)
        testvalNp = np.array(testval) / 255
        testlabelsNp = np.array(testlabels)

        nn = NeuralNetwork()

        testlabelsHot = np.array(nn.generate_labels(testlabelsNp))
        
        nn.reassign_weights(netWeights)

        _, accuracy = nn.evaluate(testvalNp, testlabelsHot)
        print("Accuracy: ", accuracy, " % ------------>", " Error: ", 100 - accuracy, " %")

main()
