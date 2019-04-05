from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import collections
import pickle
import matplotlib.pyplot as plt
import math

def get_data(data_file):
    data = pd.read_csv(data_file,header=None)
    training_data = np.array(data.values)
    labels_data = training_data[:,-1]

    encoder = OneHotEncoder(handle_unknown='ignore')   
    encoder.fit(training_data)
    total_attr = len(encoder.get_feature_names())
    
    #get unique values in desired output
    elem_freq = collections.Counter(labels_data)
    labels_count = len(list(elem_freq))
    
    features_count = total_attr - labels_count
   
    input_data = encoder.transform(training_data).toarray()
    features = input_data[:,:features_count]
    
    labels = input_data[:,features_count:]
    return features,labels

def relu(input_to_layer):
    for row in input_to_layer:
        row[row < 0] = 0
    return input_to_layer

#to avoid overflow
def sig(x):
    return 1-(1/(math.exp(x)+1)) if x<0 else 1/(math.exp(-x)+1)

def sigmoid(input_to_layer):
    vec = np.vectorize(sig)
    return vec(input_to_layer)                    
                                             
def forward_pass(input_data, activation_fn, layers):
    total_layers = len(layers)
    
    #set x0 =1 for bias
    input_data = np.c_[np.ones(input_data.shape[0]),input_data]
    layers[0]["net_input"] = input_data
    
    for i in range(total_layers-1):
        next_net_input = layers[i]["weights"] @ layers[i]["net_input"].T
        
        if activation_fn == "sigmoid":
            next_net_input = sigmoid(next_net_input)
        elif activation_fn == "relu":
            next_net_input = relu(next_net_input)
        
        #set x0=1 for next_net_input
        layers[i+1]["net_input"] = next_net_input.T
        layers[i+1]["net_input"] = np.c_[np.ones(layers[i+1]["net_input"].shape[0]),layers[i+1]["net_input"]]
    #print(sigmoid(layers[total_layers-1]["weights"] @ layers[total_layers-1]["net_input"].T ))
    

def initialize_layers(nodes_each_layer,no_of_features,batch_size):
    
    no_layers = len(nodes_each_layer)
    layers = []
    #initialize layer 0
    layer = {}
    layer["node_count"] = nodes_each_layer[0]
    layer["weights"] = np.random.normal(0, 0.001, size=(nodes_each_layer[0], no_of_features+1))
    #layer["weights"] = np.zeros((nodes_each_layer[0], no_of_features+1))
    #layer["weights"] = pickle.load(open("w_20_86","rb"))
    
    layer["net_input"] = np.zeros((batch_size,no_of_features+1))
    layers.append(layer)
    
    
    for i in range(1,no_layers):
        layer = {}
        layer["node_count"] = nodes_each_layer[i]
        layer["weights"] = np.random.normal(0, 0.001, size=(nodes_each_layer[i], nodes_each_layer[i-1]+1))
        #layer["weights"] = np.zeros((nodes_each_layer[i], nodes_each_layer[i-1]+1))
        #layer["weights"] = pickle.load(open("w_10_21","rb"))
        
        layer["net_input"] = np.zeros((batch_size,nodes_each_layer[i-1]+1))
        layers.append(layer)
    return layers


def get_derivative(z,activation_fn):
    if(activation_fn == "sigmoid"):
        return np.multiply(z,1-z)
    if(activation_fn == "relu"):
        return (z>0)*1
        
def back_propagation(labels, predictions, last_layer, layers, learning_rate, activation_fn):
    batch_size = predictions.shape[0]
    derv = get_derivative(predictions,activation_fn)
    delta_L = np.multiply(labels - predictions, derv)
    delta_w = delta_L.T @ layers[last_layer]["net_input"]
    updated_weights = layers[last_layer]["weights"] + learning_rate * delta_w/batch_size
    
    
    #back propagation in hidden layers
    for i in range(last_layer-1, -1, -1):
        y = layers[i+1]["net_input"]
        layers[i+1]["weights"] = updated_weights
        #temp = np.sum(np.multiply(updated_weights, layers[i+1]["weights"]),axis=0)
        temp = delta_L @ layers[i+1]["weights"][:,1:]
        
        derv = get_derivative(y,activation_fn)
        delta_L = np.multiply(temp, derv[:,1:])
        
        
        #remove first column as that will be all 0
        delta_L = delta_L
        
        delta_w = delta_L.T @ layers[i]["net_input"]
            
        
        updated_weights = layers[i]["weights"] + learning_rate * delta_w/batch_size
    layers[0]["weights"] = updated_weights
    
def mean_sq_error(labels, predictions):  
    #print(predictions)
    sq_diff = np.square(np.subtract(labels, predictions))
    #print(sq_diff)
    error_vector = np.sum(sq_diff,axis=1)/2
    error = np.mean(error_vector)
    return error
    

def train_per_batch(layers, features, labels, activation_fn, learning_rate):
    
    forward_pass(features, activation_fn, layers)
    
    #get the prediction of output layer
    last_layer = len(layers) - 1
    predictions = sigmoid(layers[last_layer]["net_input"] @ layers[last_layer]["weights"].T)
    
    #compute mean squared error
    error = mean_sq_error(labels, predictions)
    #print(error)
    
    #back propagate the error to update weights
    back_propagation(labels, predictions, last_layer, layers, learning_rate, activation_fn)
    return error
    

def get_accuracy(layers, features, labels, activation_fn):
    forward_pass(features, activation_fn, layers)
    
    #get the prediction
    last_layer = len(layers) - 1
    predictions = sigmoid(layers[last_layer]["net_input"] @ layers[last_layer]["weights"].T)
    
    for i in range(len(predictions)):
        max_pred = np.max(predictions[i])
        predictions[i] = np.where(predictions[i] == max_pred,1,0)
    #print(predictions)
    
    total_count = len(predictions)
    correct_count = np.sum(np.all(labels == predictions, axis=1))
    
    print("Accuracy: ", correct_count*100/total_count)


def train(nodes_each_layer, batch_size, features, labels, num_epochs, activation_fn, learning_rate, dump_file, error_metric):
    no_of_features = features.shape[1]
    layers = initialize_layers(nodes_each_layer,no_of_features, batch_size)
    for j in range(num_epochs):
        errors = []
        i = 0
        print("== EPOCH: ", j, " ==")
        while i+batch_size <= len(features):
            error = train_per_batch(layers, features[i:i+batch_size], labels[i:i+batch_size], activation_fn, learning_rate)
            errors.append(error)
            i += batch_size
            
        error_metric.append(np.mean(errors))
        
        if j % 10==0:
            features,labels = get_data("../train.csv")
            get_accuracy(layers, features, labels, "sigmoid")
    return layers

def plot(errors):
    x = np.arange(1,len(errors)+1,1)
    y = errors
    plt.plot(x, y, color='r')
    plt.title('error vs iter')
    plt.xlabel('Number of iterations')
    plt.ylabel('Avg Error')
    plt.show()

features,labels = get_data("../train.csv")
errors = []
layers = train([20,10],100,features,labels,400,"sigmoid",1,"layers.pkl",errors)

plot(errors)

