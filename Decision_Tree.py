import numpy as np
import collections
from anytree import Node, RenderTree
import pandas as pd 

def compute_entropy(labels):
    elem_freq = collections.Counter(labels)
    p1x = elem_freq[0]/labels.size
    p2x = elem_freq[1]/labels.size
    if(p1x == 1 or p2x == 1):
        return 0
    H_X = -1 * (p1x * np.log2(p1x) + p2x * np.log2(p2x))
    return H_X
  
def info_gain(H_parent, features, labels, feature_no):
    #arr = np.array(["weak","strong","weak","strong","mild","mild","weak","strong","mild","strong","strong","weak"])
    #labels = np.array([1,0,1,0,0,0,1,1,1,0,1,1])
    feature = features[:,feature_no]
    elem_freq = collections.Counter(feature)
    attr_vals = list(elem_freq)
    #print(attrs)
    prob_vals = np.array([elem_freq[val]/feature.size for val in attr_vals])
    h_vals = []
    for val in attr_vals:
        val_labels = labels[np.where(feature == val)]
        h_vals.append(compute_entropy(val_labels))

    IG = H_parent - np.sum(np.multiply(prob_vals,h_vals))  
    return IG

def break_data(features, labels, feature_no):
    feature = features[:,feature_no]
    #print(feature)
    elem_freq = collections.Counter(feature)
    attr_vals = list(elem_freq)
    
    features_set = [features[np.where(feature == val),:][0] for val in attr_vals]
    labels_set = [labels[np.where(feature == val)] for val in attr_vals]
    return attr_vals,features_set,labels_set

def partition_data(features, labels, feature_no):
    feature = features[:,feature_no]
    print(feature)
    median = np.median(feature)
    features_set = []
    labels_set = []
    #check if only 2 values are there in continuous data
    elem_freq = collections.Counter(feature)
    attr_vals = np.array(list(elem_freq))
    if(attr_vals.size == 2):
        val = np.min(attr_vals)
        features_set.append(features[np.where(feature == val)])
        features_set.append(features[np.where(feature != val)])
        labels_set.append(labels[np.where(feature == val)])
        labels_set.append(labels[np.where(feature !=val)])
    else:
        features_set.append(features[np.where(feature <= median)])
        features_set.append(features[np.where(feature > median)])
        labels_set.append(labels[np.where(feature <= median)])
        labels_set.append(labels[np.where(feature > median)])
    return median,features_set,labels_set
def preprocess_continuous_attr(features, feature_no):
    feature = features[:, feature_no].astype(int)
    
    #check if it has only two values
    elem_freq = collections.Counter(feature)
    attr_vals = list(elem_freq)
    if(len(attr_vals) == 2):
        val = attr_vals[0]
        features[:, feature_no] = np.where(feature == val, 0, 1)
    
    else:
        median = np.median(feature)
        features[:, feature_no] = np.where(feature <= median, 0, 1)
    return features

def get_training_data(file):
    data = pd.read_csv(file)
    training_data = np.array(data.values[1:,1:]).astype(int)
    features = training_data[:,:-1]
    labels = training_data[:,-1]
    return features, labels
    
def preprocess_data(file):
    data = pd.read_csv(file)
    training_data = np.array(data.values[1:,1:]).astype(int)
    #print(training_data)
    continuous_attr = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    for feature_no in continuous_attr:
        training_data = preprocess_continuous_attr(training_data, feature_no)
    features = training_data[:,:-1]
    labels = training_data[:,-1]
    return features, labels

class NodeType:

    node_count = 0
    def __init__(self, data, majority):
        self.data = data
        self.majority = majority
        self.children = None
        self.median = None
        NodeType.node_count += 1

    def Print(self,tree_node):
        print(self.data)
        print(self.children)
        if self.children:
            for child in self.children.keys():
                temp_tree_node = Node(str(self.children[child].data),parent=tree_node)
                self.children[child].Print(temp_tree_node)
    
continuous_attr = {0, 1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
def grow_tree(attr_set, features, labels, que_part):
    #compute entropy of node
    entropy = compute_entropy(labels)
    
    #leaf node if all examples are either true of false
    if entropy == 0:
        return NodeType(None,labels[0])
    
    #get the majority labels
    elem_freq = collections.Counter(labels)
    majority = elem_freq.most_common(1)[0][0]
    
    #if we ran out of attributes
    if not attr_set:
        return NodeType(None,majority)
              
    max_gain = -1
    max_gain_feature = list(attr_set)[0] 
    
    processed_features = np.copy(features)
    if(que_part == 'c'):
        #for continuous data modify data based on median
        for attr_no in continuous_attr:
            processed_features = preprocess_continuous_attr(processed_features, attr_no)
    
    #get the best attribute
    for feature_no in attr_set:
        IG = info_gain(entropy, processed_features, labels, feature_no)
        print("IG:",IG,":",feature_no)
       
        if(IG > max_gain):
            max_gain = IG
            max_gain_feature = feature_no
            
    #stop if IG is very less
    if max_gain <= 1e-10:
        return NodeType(None,majority)
    
   #create node with feature_no,majority
    node = NodeType(max_gain_feature,majority)
    node.children = {}
    
    #partition based on the best feature column
    if(que_part == 'c' and (max_gain_feature in continuous_attr)):
        median,features_set,labels_set = partition_data(features, labels, max_gain_feature)
        node.median = median
        attr_vals = [0,1]
    elif(que_part == 'c'):
        attr_vals,features_set,labels_set = break_data(features, labels, max_gain_feature)
    else:
        attr_vals,features_set,labels_set = break_data(features, labels, max_gain_feature)
        attr_set = attr_set - {max_gain_feature}
    print(max_gain_feature)
    
    for i in range(len(attr_vals)):
        val = attr_vals[i]
        node.children[val] = grow_tree(attr_set, features_set[i], labels_set[i],que_part)
        
    return node

