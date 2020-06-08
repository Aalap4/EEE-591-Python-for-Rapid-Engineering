# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:18:53 2020

@author: Aalap
"""

#Aalap Paragbhai Doshi
#Project 2
#ASU ID 1217130629

import numpy as np                                     # needed for arrays
import pandas as pd                                    # data frame
import matplotlib.pyplot as plt                        # modifying plot
from sklearn.model_selection import train_test_split   # splitting data
from sklearn.preprocessing import StandardScaler       # scaling data
from sklearn.linear_model import LogisticRegression    # learning algorithm
from sklearn.decomposition import PCA                  # PCA package
from sklearn.metrics import accuracy_score             # grading
from sklearn.neural_network import MLPClassifier       # Classifier
from sklearn.model_selection import GridSearchCV       #GridSearch to Find the Best Parameters
from warnings import filterwarnings                    #Ignore the warnings
from sklearn.metrics import confusion_matrix           #plot confusion matrix

filterwarnings('ignore')

# read the database. Since it lackets headers, put them in
df_sonar = pd.read_csv('sonar_all_data_2.csv',header=None)           # read the csv file

# list out the labels

N = len(df_sonar.columns)                                         # Returns the number of columns in the dataset
X = df_sonar.iloc[:,0:N-2].values                                 # features are in columns 0:(N-2) 
y = df_sonar.iloc[:,-1].values                                    # classes are in column 0!


# now split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

stdsc = StandardScaler()                                          # apply standardization
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test) 


acc=[];                                                           #empty list to store accuracy
y_pred_list=[]                                                    #empty list to store y_pred values

####################################################
#To perform PCA on the dataset,plot the components vs accuracy graph.
####################################################

for i in range(1,61):
    
    pca = PCA(n_components=i)  
    print('Number of Components: ',i,'\n')                            # only keep two "best" features!
    X_train_pca = pca.fit_transform(X_train_std)                      # apply to the train data
    X_test_pca = pca.transform(X_test_std)                            # do the same to the test data
    model = MLPClassifier( hidden_layer_sizes=(100,), activation='logistic', batch_size=20, max_iter=500,
                      solver='lbfgs', learning_rate= 'adaptive', alpha=0.00001, random_state=0)
   
    model.fit(X_train_pca,y_train)                            #fitting the best_parameter space 
    y_pred = model.predict(X_test_pca)                        # how we did on the test data?
    y_pred_list.append(y_pred)                                #predictions appended to a list
    
    print('Number in test ',len(y_test))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    
    print("\n#################################################################\n")
    #X_comb_pca = np.vstack((X_train_pca, X_test_pca))
    #y_comb = np.hstack((y_train, y_test))
    #print('Number in combined ',len(y_comb))
    #y_comb_pred = model.predict(X_comb_pca)
    #print('Misclassified combined samples: %d' % (y_comb != y_comb_pred).sum())
    #print('Combined Accuracy: %.2f' % accuracy_score(y_comb, y_comb_pred),'\n')

    acc.append(accuracy_score(y_test, y_pred))        #a small logic used to select the index of the maximum accuracy
    Max_test_acc=max(acc)                             #and also the minimum component which gives that accuracy out of may be 2 or 3 others
    temp = 0
    index=[]
    for i in range(1,len(acc)):
        temp = acc[i]
        if (temp==Max_test_acc):
            index.append(i+1)
           

print('The maximum accuracy: ', Max_test_acc)
print('Number of components that achieved maximum accuracy: ',min(index))

array=np.arange(1, N-1, 1)                        #np array of consisting of 1-60 components
plt.plot(array, acc)
plt.xlabel("Number of PCA components")
plt.ylabel("Accuracy")


###############################################
#To create a confusion matrix to detect false
#positives and false negatives.
###############################################

#In case there are two maximum values , choose the minimum number of components so as to achieve minimal complexity
max_values_minimum=min(index)                  

#Confusion Matrix
confusion_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred_list[max_values_minimum-1], labels=['R', 'M']),
index=['true:R', 'true:M'],
columns=['pred:R', 'pred:M'])
print('Confusion Matrix:',confusion_matrix)

    
plt.show()
    
    
    

# Grid Search for Algorithm Tuning
'''
alphas = {'hidden_layer_sizes':[100,200,500], 'activation':['tanh','relu','logistic'], 'batch_size':[20,100], 'max_iter':[2000,1000], 'alpha':[0.00001, 0.001]
                      ,'solver':['sgd','adam'], 'tol':[0.0001,0.001], 'learning_rate':['adaptive', 'invscaling'] }

# create and fit a ridge regression model, testing each alpha
grid = GridSearchCV(estimator=MLPClassifier(), param_grid=alphas)
grid.fit(X_train_pca,y_train)
print(grid)

# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_params_)
'''
