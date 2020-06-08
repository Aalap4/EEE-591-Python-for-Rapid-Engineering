#!/usr/bin/env python
# coding: utf-8

# In[3]:



#Python Project 1 Funny Money

#Aalap Paragbhai Doshi
#ASU ID- 1217130629
#Took Reference of Online resources and also code available on Canvas modules

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data
import matplotlib.pyplot as plt         # used for plotting
from matplotlib import cm as cm         # for the color map
import seaborn as sns                   # data visualization


def mosthighlycorrelated(mydataframe, numtoreport): 
    cormatrix = mydataframe.corr()                      # find the correlations 

    
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T 

   
    cormatrix = cormatrix.stack()     # rearrange so the reindex will work...
    
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index() 
    
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"] 
    print("\nMost Highly Correlated")
    print(cormatrix.head(numtoreport))     # print the top values



def correl_matrix(X):
    # create a figure that's 7x7 (inches?) with 100 dots per inch
    fig = plt.figure(figsize=(7,7), dpi=100)

    # add a subplot that has 1 row, 1 column, and is the first subplot
    ax1 = fig.add_subplot(111)

    # get the 'jet' color map
    cmap = cm.get_cmap('jet',30)

    # Perform the correlation and take the absolute value of it. Then map
    # the values to the color map using the "nearest" value
    cax = ax1.imshow(np.abs(X.corr()),interpolation='nearest',cmap=cmap)

    # now set up the axes
    major_ticks = np.arange(0,len(X.columns),1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True,which='both',axis='both')
    plt.title('Correlation Matrix')
    ax1.set_xticklabels(X.columns,fontsize=9)
    ax1.set_yticklabels(X.columns,fontsize=12)

    # add the legend and show the plot
    fig.colorbar(cax, ticks=[-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()



def pairplotting(df):
    sns.set(style='whitegrid', context='notebook')   # set the apearance
    sns.pairplot(df,height=2.5)                      # create the pair plots
    plt.show()                                       # and show them


note = pd.read_csv('data_banknote_authentication.csv',header=None)
note.columns=['variance','skewness','curtosis','entropy','class']
print('first 5 observations',note.head(5))



#  descriptive statistics
print('\nDescriptive Statistics')
print(note.describe())

mosthighlycorrelated(note,5)                # generate most highly correlated list
correl_matrix(note)                         # generate the covariance heat plot
pairplotting(note)                          # generate the pair plot


# In[ ]:




