import numpy as np
import collections
from anytree import Node, RenderTree
import pandas as pd 

#computes entropy at a given node
def compute_entropy(labels):
    elem_freq = collections.Counter(labels)
    p1x = elem_freq[0]/labels.size
    p2x = elem_freq[1]/labels.size
    if(p1x == 1 or p2x == 1):
        return 0
    H_X = -1 * (p1x * np.log2(p1x) + p2x * np.log2(p2x))
    return H_X
  
#information gain wrt given attribute (feature_no)
def info_gain(H_parent, features, labels, feature_no):
    #arr = np.array(["weak","strong","weak","strong","mild","mild","weak","strong","mild","strong","strong","weak"])
    #labels = np.array([1,0,1,0,0,0,1,1,1,0,1,1])
    feature = features[:,feature_no]
    elem_freq = collections.Counter(feature)
    attr_vals = list(elem_freq)
    #print(attr_vals)
    prob_vals = np.array([elem_freq[val]/feature.size for val in attr_vals])
    h_vals = []

    #entropy of every possible value in the attribute
    for val in attr_vals:
        val_labels = labels[np.where(feature == val)]
        h_vals.append(compute_entropy(val_labels))

    IG = H_parent - np.sum(np.multiply(prob_vals,h_vals))  
    return IG

#partitioning the data based on attribute for non continuous data
def break_data(features, labels, feature_no):
    feature = features[:,feature_no]
    #print(feature)
    elem_freq = collections.Counter(feature)
    attr_vals = list(elem_freq)
    
    features_set = [features[np.where(feature == val),:][0] for val in attr_vals]
    labels_set = [labels[np.where(feature == val)] for val in attr_vals]
    return attr_vals,features_set,labels_set

#partitioning the data based on attribute for continuous data
def partition_data(features, labels, feature_no):
    feature = features[:,feature_no]
    #print(feature)
    median = np.median(feature)

    #check if only 2 values are there in continuous data
    elem_freq = collections.Counter(feature)
    attr_vals = np.array(list(elem_freq))
    if(attr_vals.size == 2):
        attr_vals = np.sort(attr_vals)
        features_set = [features[np.where(feature == val),:][0] for val in attr_vals]
        labels_set = [labels[np.where(feature == val)] for val in attr_vals]
    else:
        features_set = []
        labels_set = []
        features_set.append(features[np.where(feature <= median)])
        features_set.append(features[np.where(feature > median)])
        labels_set.append(labels[np.where(feature <= median)])
        labels_set.append(labels[np.where(feature > median)])
    return attr_vals = median,features_set,labels_set

#convert continuous data into binary data
def preprocess_continuous_attr(features, feature_no):
    feature = features[:, feature_no].astype(int)
    #check if it has only two values
    elem_freq = collections.Counter(feature)
    attr_vals = list(elem_freq)

    #check if only 2 values are there in continuous data
    if(len(attr_vals) == 2):
        val = attr_vals[0]
        features[:, feature_no] = np.where(feature == val, 0, 1)
    else:
        median = np.median(feature)
        features[:, feature_no] = np.where(feature <= median, 0, 1)
    return features

def get_training_data(data_file):
    data = pd.read_csv(data_file)
    training_data = np.array(data.values[1:,1:]).astype(int)
    features = training_data[:,:-1]
    labels = training_data[:,-1]
    return features, labels

#preprocess continuous attributes    
def preprocess_data(data_file):
    features, labels = get_training_data(data_file)
    #print(training_data)
    continuous_attr = {0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
    for feature_no in continuous_attr:
        features = preprocess_continuous_attr(features, feature_no)
    
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
    
continuous_attr = {0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
binary_attr = {1}

#creating decision tree
def grow_tree(attr_set, features, labels, que_part):
    #compute entropy of node
    entropy = compute_entropy(labels)
    
    #leaf node if all examples are either true of false
    if entropy == 0:
        return NodeType(None,labels[0])
    
    #get the majority labels
    elem_freq = collections.Counter(labels)
    majority = elem_freq.most_common(1)[0][0]
    
    #if we ran out of attributes for que part (a)
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
        if(IG > max_gain):
            max_gain = IG
            max_gain_feature = feature_no
    
    print("max IG:",max_gain,",attr:",max_gain_feature)
    #stop if max IG is very less
    if max_gain <= 1e-10:
        return NodeType(None,majority)
    
   #create node with feature_no,majority
    node = NodeType(max_gain_feature,majority)
    node.children = {}
    
    #partition based on the best feature column
    if(que_part == 'c' and (max_gain_feature not in binary_attr)):
        median,features_set,labels_set = partition_data(features, labels, max_gain_feature)
        node.median = median
        attr_vals = [0,1]
    elif(que_part == 'c'):
        attr_vals,features_set,labels_set = break_data(features, labels, max_gain_feature)
    else:
        attr_vals,features_set,labels_set = break_data(features, labels, max_gain_feature)
        attr_set = attr_set - {max_gain_feature}
    
    for i in range(len(attr_vals)):
        val = attr_vals[i]
        node.children[val] = grow_tree(attr_set, features_set[i], labels_set[i],que_part)
        
    return node

def get_accuracy(features,label,root,que_part):
    output = []
    for row in features:
        node = root
        while node.children:
            attr = node.data
            attrval = row[attr]
            if (attr not in binary_attr) and que_part == 'c':
                if attrval <= node.median:
                    node = node.children[0]
                else:
                    node = node.children[1]
            elif attrval not in node.children:
                break
            else:
                node = node.children[attrval]
        output.append(node.majority)
    predictions = np.array(output)
    correct_count = np.sum([predictions==labels])
    return (correct_count*100)/labels.size

def decision_tree(que_part, train_file, test_file, validation_file):
    if(que_part == 'a'):
        features,labels = preprocess_data(train_file)             
        root = grow_tree(set(range(23)), features, labels, que_part)
        #for graphical view
        tree_node = Node(str(root.data))
        #print(root.Print(tree_node))
        #get Accuracy
        features,labels = preprocess_data(test_file)  
        acc = get_accuracy(features,label,root,que_part)
        print(acc)
    elif(que_part == 'c'):
        features,labels = get_training_data(train_file)             
        root = grow_tree(set(range(23)), features, labels, que_part)
        tree_node = Node(str(root.data))
        #for graphical view
        tree_node = Node(str(root.data))
        #print(root.Print(tree_node))
        #get Accuracy
        features,labels = get_training_data(test_file)  
        acc = get_accuracy(features,label,root,que_part)
        print(acc)


