import numpy as np
import collections
from anytree import Node, RenderTree
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import queue

#fetch the data from the file
def get_data(data_file):
    data = pd.read_csv(data_file)
    training_data = np.array(data.values[1:,1:]).astype(int)
    features = training_data[:,:-1]
    labels = training_data[:,-1]
    print(features.shape,labels.shape)
    return features, labels

#one hot encoder used for changing the categorical attributes to binary attributes
def one_hot_encoder(features,cols,col_values):
    cols_to_be_left = list(set(range(features.shape[1])) - set(cols))
    features_processed = features[:, cols_to_be_left]
    count_rows = features.shape[0]
    for col, values in zip(cols, col_values):
        val_to_int = {val: idx for idx, val in enumerate(values)}
        new_cols = np.zeros((count_rows, len(values)))
        for row in range(count_rows):
            value = features[row, col]
            new_cols[row, val_to_int[value]] = 1
        features_processed = np.hstack((features_processed, new_cols))
    return features_processed

#convert continuous data into binary data
def preprocess_continuous_attr(features, feature_no):
    feature = features[:, feature_no].astype(int)    
    median = np.median(feature)
    features[:, feature_no] = np.where(feature <= median, 0, 1)
    return features

#preprocess continuous attributes    
def preprocess_data(data_file):
    features, labels = get_data(data_file)
    #print(training_data)
    continuous_attr = {0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
    for feature_no in continuous_attr:
        features = preprocess_continuous_attr(features, feature_no)
    return features, labels

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

    #this is binary partition based on median
    features_set = []
    labels_set = []
    features_set.append(features[np.where(feature <= median)])
    features_set.append(features[np.where(feature > median)])
    labels_set.append(labels[np.where(feature <= median)])
    labels_set.append(labels[np.where(feature > median)])
    return median,features_set,labels_set

# a class defined for managing the nodes in the decision tree
class NodeType:
    #total no of nodes in the decision tree
    node_count = 0
    #list of all node objects in the decision tree
    node_list = []
    def __init__(self, data, majority):
        self.data = data
        self.majority = majority
        self.children = None
        self.median = None
        NodeType.node_count += 1
    
    def BFS_traversal(self):
        nodes = queue.Queue(NodeType.node_count)
        NodeType.node_list = []
        nodes.put(self)
        while(not nodes.empty()):
            node = nodes.get()
            if node.children:
                for child in node.children.values():
                    nodes.put(child)
            NodeType.node_list.append(node)

    def Print(self,tree_node):
        print(self.data)
        print(self.children)
        if self.children:
            for child in self.children.keys():
                temp_tree_node = Node(str(self.children[child].data),parent=tree_node)
                self.children[child].Print(temp_tree_node)
    
continuous_attr = {0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}

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

#creating decision tree
def grow_tree(attr_set, features, labels, setting="median_fixed"):
    #compute entropy of node
    entropy = compute_entropy(labels)
    
    #leaf node if all examples are either true of false
    if entropy == 0:
        return NodeType(None,labels[0])
    
    #get the majority labels
    elem_freq = collections.Counter(labels)
    majority = elem_freq.most_common(1)[0][0]
    
    max_gain = -1
    max_gain_feature = list(attr_set)[0] 
    processed_features = np.copy(features)

    #preprocess at each node in this setting
    if(setting == "median_variable"):
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
    if(setting == "median_variable" and (max_gain_feature in continuous_attr)):
        median,features_set,labels_set = partition_data(features, labels, max_gain_feature)
        node.median = median
        attr_vals = [0,1]
    else:
        attr_vals,features_set,labels_set = break_data(features, labels, max_gain_feature)
    
    for i in range(len(attr_vals)):
        val = attr_vals[i]
        node.children[val] = grow_tree(attr_set, features_set[i], labels_set[i],setting)
        
    return node

def get_accuracy(features,labels,root,setting = "median_fixed"):
    output = []
    for row in features:
        node = root
        while node.children:
            attr = node.data
            attrval = row[attr]
            if (attr in continuous_attr) and setting == "median_variable":
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

def part_a(train_file, test_file, val_file, setting = "median_fixed"):
    train_features,train_labels = preprocess_data(train_file)
    val_features,val_labels = preprocess_data(val_file)      
    test_features,test_labels = preprocess_data(test_file)

    root = grow_tree(set(range(23)), train_features, train_labels, setting)

    #for graphical view
    #tree_node = Node(str(root.data))
    #print(root.Print(tree_node))
    
    #get Accuracy 
    train_acc = get_accuracy(train_features,train_labels,root,setting)
    print("Training set Accuracy:",train_acc)

    val_acc = get_accuracy(val_features,val_labels,root,setting)
    print("Validation set Accuracy:",val_acc)

    test_acc = get_accuracy(test_features,test_labels,root,setting)
    print("Testing set Accuracy:",test_acc)

def tree_pruning(list_nodes,val_features,val_labels,root):
    prev_acc = get_accuracy(val_features,val_labels,root)
    iter = 0
    while(iter <= 100000):
        accuracies = []
        for node in list_nodes:
            temp_child = node.children
            node.children = {}
            accuracies.append(get_accuracy(val_features,val_labels,root))
            node.children = temp_child
            
        next_acc = max(accuracies)
        print("iteration:",iter,",Max_Accuracy",next_acc)
        node_to_prune = list_nodes[accuracies.index(next_acc)]
        if((next_acc - prev_acc) <= 1e-4):
            break
        prev_acc = next_acc
        node_to_prune.children = {}
        iter += 1
    return prev_acc

def part_b(train_file, test_file, val_file):
    train_features,train_labels = preprocess_data(train_file)
    val_features,val_labels = preprocess_data(val_file)      
    test_features,test_labels = preprocess_data(test_file)

    root = grow_tree(set(range(23)), train_features, train_labels)
    root.BFS_traversal()  
    max_acc = tree_pruning(NodeType.node_list,val_features,val_labels,root)
    print("Accuracy after pruning in validation set:", max_acc)
    
    #Tree has been pruned, now get the Accuracies 
    train_acc = get_accuracy(train_features,train_labels,root)
    print("Training set Accuracy:",train_acc)

    val_acc = get_accuracy(val_features,val_labels,root)
    print("Validation set Accuracy:",val_acc)

    test_acc = get_accuracy(test_features,test_labels,root)
    print("Testing set Accuracy:",test_acc)


#---------------------------part (d)-----------------------------------------------

def get_acc_using_params(flag,params,train_features,train_labels,val_features,val_labels):   
    if(flag == 0):
        dt = DecisionTreeClassifier(criterion=params[0],random_state=params[1])
    if(flag == 1):
        dt = DecisionTreeClassifier(criterion=params[0],max_depth=params[2],random_state=params[1])
    elif(flag == 2):
        dt = DecisionTreeClassifier(criterion=params[0],min_samples_split=params[2],random_state=params[1])
    elif(flag == 3):
        dt = DecisionTreeClassifier(criterion=params[0],min_samples_leaf=params[2],random_state=params[1])
    elif(flag == 4):
        dt = DecisionTreeClassifier(criterion=params[0],min_samples_split = params[2], min_samples_leaf=params[3],random_state=params[1])

    dt.fit(train_features,train_labels)
    val_accuracy = dt.score(val_features,val_labels)
    return val_accuracy

def plot_acc(x_vals, accuracy):
    max_acc=max(accuracy)
    plt.plot(x_vals, accuracy)
    
    plt.legend(['Max Accuracy: %.1f' % max_acc])
    plt.ylabel('Accuracy-->')
    plt.xlabel('parameter------>')
    
    plt.show()
    plt.close()
    
def part_d(train_file, test_file, val_file):
    train_features,train_labels = get_data(train_file)
    val_features,val_labels = get_data(val_file)
    params = ["gini", 0]
    val_accuracy = get_acc_using_params(0,params,train_features,train_labels,val_features,val_labels)

    print("Validation set Accuracy:",val_accuracy*100)
   
    print("1.varying max_depth ....")
    depths = list(range(1,50))
    accuracy=[]
    for d in depths:
        params = ["gini", 0, d]
        val_accuracy = get_acc_using_params(1,params,train_features,train_labels,val_features,val_labels)
        accuracy.append(val_accuracy*100)
        print("depth:",d)
    plot_acc(depths, accuracy)
    
    print("2.Varying min_samples_split")
    split_sizes=list(range(5, 200, 20))
    accuracy=[]
    for split in split_sizes:
        params = ["gini",0,split]
        val_accuracy = get_acc_using_params(2,params,train_features,train_labels,val_features,val_labels)
        accuracy.append(val_accuracy*100)
        print("split_size:",split)
    plot_acc(split_sizes, accuracy)
    
    print("3.Varying min_samples_leaf")
    leaf_sizes=list(range(10, 200, 5))
    accuracy=[]
    for leaf in leaf_sizes:
        params = ["gini",0,leaf]
        val_accuracy = get_acc_using_params(3,params,train_features,train_labels,val_features,val_labels)
        accuracy.append(val_accuracy*100)
        
    plot_acc(leaf_sizes, accuracy)

    print("4.Varying both min_samples_leaf and min_samples_split")
    accuracy=[]
    for split in split_sizes:
         for leaf in leaf_sizes:
            params = ["gini",0,split,leaf]
            val_accuracy = get_acc_using_params(4,params,train_features,train_labels,val_features,val_labels)
            accuracy.append(val_accuracy*100)
            print("split,leaf:",split,",",leaf)
    x_vals = list(range(len(accuracy)))    
    plot_acc(x_vals, accuracy)


def part_e(train_file, test_file, val_file):
    train_features,train_labels = get_data(train_file)
    val_features,val_labels = get_data(val_file)
    cols = [1,2,3,5,6,7,8,9,10]
    col_vals = [[1, 2],[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3],[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    train_features = one_hot_encoder(train_features,cols,col_vals)
    val_features = one_hot_encoder(val_features,cols,col_vals)
    for leaf in list(range(60,100,5)):
        params = ["gini",0,leaf]
        val_accuracy = get_acc_using_params(3,params,train_features,train_labels,val_features,val_labels)
        print(leaf,":",val_accuracy*100)
    #part_d(train_features,train_labels, val_features,val_labels)

def part_f(train_file, test_file, val_file):
    train_features,train_labels = get_data(train_file)
    val_features,val_labels = get_data(val_file)
    cols = [1,2,3,5,6,7,8,9,10]
    col_vals = [[1, 2],[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3],[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    train_features = one_hot_encoder(train_features,cols,col_vals)
    val_features = one_hot_encoder(val_features,cols,col_vals)

    rf = RandomForestClassifier(criterion="gini",random_state=0)
    rf.fit(train_features,train_labels)
   
    acc = rf.score(val_features,val_labels)
    print("Validation set Accuracy:",acc*100)
   
    print("1.varying no of trees ....")
    trees = list(range(10,200,10))
    accuracy=[]
    for t in trees:
        rf = RandomForestClassifier(criterion="entropy",random_state=0,n_estimators=t)
        rf.fit(train_features,train_labels)
   
        acc = rf.score(val_features,val_labels)
        print("Validation set Accuracy:",acc*100)
        accuracy.append(acc*100)
        print("tree_count:",t)
    plot_acc(trees, accuracy)
    accuracy=[]
    for t in trees:
        rf = RandomForestClassifier(criterion="entropy",random_state=0,n_estimators=t,bootstrap=False)
        rf.fit(train_features,train_labels)
   
        acc = rf.score(val_features,val_labels)
        print("Validation set Accuracy:",acc*100)
        accuracy.append(acc*100)
        print("tree_count:",t)
    plot_acc(trees, accuracy)
    
    print("2.Varying max_features.")
    feature_sizes=list(range(1, 10))
    accuracy = []
    for feature in feature_sizes:
        rf = RandomForestClassifier(criterion="entropy",random_state=0,max_features=feature)
        rf.fit(train_features,train_labels)
   
        acc = rf.score(val_features,val_labels)
        print("Validation set Accuracy:",acc*100)
        accuracy.append(acc*100)
        print("max_features:",feature)
    plot_acc(feature_sizes, accuracy)
    accuracy = []
    for feature in feature_sizes:
        rf = RandomForestClassifier(criterion="entropy",random_state=0,max_features=feature,bootstrap=False)
        rf.fit(train_features,train_labels)
   
        acc = rf.score(val_features,val_labels)
        print("Validation set Accuracy:",acc*100)
        accuracy.append(acc*100)
        print("max_features:",feature)
    plot_acc(feature_sizes, accuracy)


def decision_tree(sub_part, train_file, test_file, val_file):
    if(sub_part == 1):
        part_a(train_file, test_file, val_file)
    elif(sub_part == 2):
        part_b(train_file, test_file, val_file)
    elif(sub_part == 3):
        part_a(train_file, test_file, val_file, setting = "median_variable")
    elif(sub_part == 4):
        part_d(train_file, test_file, val_file)
    elif(sub_part == 5):
        part_e(train_file, test_file, val_file)
    elif(sub_part == 6):
        part_f(train_file, test_file, val_file)

#decision_tree('c',"../credit-cards.train.csv","../credit-cards.train.csv","../test.csv")
#part_b("../credit-cards.train.csv","../credit-cards.val.csv")
#train_features,train_labels = get_data("../credit-cards.train.csv")
#val_features,val_labels = get_data("../credit-cards.val.csv")
#part_d(train_features,train_labels, val_features,val_labels)
part_f("../credit-cards.train.csv","../credit-cards.test.csv","../credit-cards.val.csv")




