#!/usr/bin/env python
# coding: utf-8

# In[29]:



#Python Project 1 Funny Money

#Aalap Paragbhai Doshi
#ASU ID- 1217130629
#Took Reference of Online resources and also code available on Canvas modules


import numpy as np  
import pandas as pd                                   
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def metrics(y_test,y_pred,y_combined,y_combined_pred) :
   
    print('Misclassified samples: %d' % (y_test != y_pred).sum()) 
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    
    #return accuracy_score(y_test, y_pred,y_combined,y_combined_pred)

def stdize(train_x,test_x) :
    
    sc = StandardScaler()                      # create the standard scalar
    sc.fit(train_x)                            # compute the required transformation
    train_x_std = sc.transform(train_x)        # apply to the training data
    test_x_std = sc.transform(test_x)          # and SAME transformation of test data!!!
    X_combined_std = np.vstack((train_x_std,test_x_std))
    X_combined = np.vstack((train_x, test_x))

    return train_x_std,test_x_std,X_combined_std,X_combined

def perceptron(train_x,test_x,train_y,test_y) :
    
    train_x_std,test_x_std,X_combined_std,X_combined = stdize(train_x,test_x)
    ppn = Perceptron(max_iter=10, tol=1e-3, eta0=1e-3, fit_intercept=True, random_state=0, verbose=False)
    ppn.fit(train_x_std, train_y.values.ravel())              # do the training
    y_pred = ppn.predict(test_x_std)                  # predict the output
    y_combined_pred = ppn.predict(X_combined_std)
    

    return y_pred,y_combined_pred

def logistic_regression(train_x,test_x,train_y,test_y) :
    
    
    train_x_std,test_x_std,X_combined_std,X_combined = stdize(train_x,test_x)

    lr = LogisticRegression(C=10.0, solver='lbfgs', multi_class='ovr', random_state=1, class_weight='balanced')
    lr.fit(train_x_std, train_y)                # apply the algorithm to training data
    y_pred = lr.predict(test_x_std)             # predict the output
    y_combined_pred = lr.predict(X_combined_std)
    
    return y_pred,y_combined_pred
    
def SVM(train_x,test_x,train_y,test_y) :
    
    train_x_std,test_x_std,X_combined_std,X_combined = stdize(train_x,test_x)
   
    svm = SVC(kernel='linear', C=3.0, random_state=0)
    svm.fit(train_x_std, train_y)                        # do the training
    y_pred = svm.predict(test_x_std)                     # predict the output
    y_combined_pred = svm.predict(X_combined_std)
    

    return y_pred,y_combined_pred
def decision_tree(train_x,test_x,train_y,test_y):
    
    train_x_std,test_x_std,X_combined_std,X_combined = stdize(train_x,test_x)
    
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=0)
    tree.fit(train_x,train_y)                       # apply the algorithm to training data
    y_pred = tree.predict(test_x)                   # predict the output
    y_combined_pred = tree.predict(X_combined)
   
    return y_pred,y_combined_pred

def random_forest(train_x,test_x,train_y,test_y) :
    
    train_x_std,test_x_std,X_combined_std,X_combined = stdize(train_x,test_x)
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10,random_state=1,min_samples_split = 5, n_jobs=2)
    forest.fit(train_x,train_y)             # apply the algorithm to training data
  
    y_pred = forest.predict(test_x)         # see how we do on the test data
    y_combined_pred = forest.predict(X_combined)
   
    return y_pred,y_combined_pred


def knn(train_x,test_x,train_y,test_y):
    
    
    train_x_std,test_x_std,X_combined_std,X_combined = stdize(train_x,test_x)
    
    knn = KNeighborsClassifier(n_neighbors=24,algorithm='auto',p=2,metric='minkowski')
    knn.fit(train_x_std,train_y)                    # apply the algorithm to training data
    
    y_pred = knn.predict(test_x_std)                # predict the output
    y_combined_pred = knn.predict(X_combined_std)
    
    return y_pred,y_combined_pred
    

# Function to run each of the algorithm

def algorithms(algo,train_x,test_x,train_y,test_y) :
    if(algo == 'prp') :
        print('\t\b PERCEPTRON \t')
        pred_y,y_combined_pred = perceptron(train_x,test_x,train_y,test_y)
    if(algo == 'lor') :
        print('\t\b LOGISTIC REGRESSION \t')
        pred_y,y_combined_pred  = logistic_regression(train_x,test_x,train_y,test_y)
    if(algo == 'svm') :
        print('\t\b SUPPORT VECTOR MACHINES \t')
        pred_y,y_combined_pred  = SVM(train_x,test_x,train_y,test_y)
    if(algo == 'dtr') :
        print('\t\b DECISION TREES \t')
        pred_y,y_combined_pred  = decision_tree(train_x,test_x,train_y,test_y,)
    if(algo == 'raf') :
        print('\t\b RANDOM FORESTS \t')
        pred_y,y_combined_pred  = random_forest(train_x,test_x,train_y,test_y)
    if(algo == 'knn') :
        print('\t\b K NEAREST NEIGHBORS \t')
        pred_y,y_combined_pred  = knn(train_x,test_x,train_y,test_y)
        
    return pred_y,y_combined_pred


filename = 'data_banknote_authentication.csv'
data = pd.read_csv(filename)
df = pd.DataFrame(data)                     #DataFrame of the data
X = df.iloc[:,0:3]                          #selecting all of the features
Y = df.iloc[:,4]                            #the target value


train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.3,random_state=0) #splitting the data into train and test
methods = ['prp','lor','svm','dtr','raf','knn']                                    #set of algorithms to be run

for method in methods:
    y_predicted,y_combined_pred = algorithms(method,train_x,test_x,train_y,test_y)
    y_combined = np.hstack((train_y, test_y))
    metrics(test_y,y_predicted,y_combined,y_combined_pred)
    print('\n\n')
    


# In[ ]:




