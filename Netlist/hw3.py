#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Aalap Paragbhai Doshi



import numpy as np                     # needed for arrays
from numpy.linalg import solve,det         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants



def ranknetlist(netlist):              # pass in the netlist

    nodes= [] # storing the nodes as an list 
    max_node = 0 #number of nodes in the circuit
    for comp in netlist :
        if (comp[COMP.TYPE] == COMP.R) : # if the component is a resistor , append the positions between which it is present
            i =  comp[COMP.I]-1
            j =  comp[COMP.J]-1
            if (i >= 0):
            #checking if the node is already present in the list
                if ( i not in nodes ) :
                    nodes.append(i)
                    max_node += 1
            if (j >= 0):
                if ( j not in nodes ) :
                    nodes.append(j)
                    max_node += 1
        elif (comp[COMP.TYPE] == COMP.VS) : # if it is a voltage source , add 1 to row and columns .
            max_node += 1
    
    
    return nodes,max_node

############################################################
# Function to stamp the components into the netlist        #
############################################################

def stamper(y_add,netlist,currents,voltages,num_nodes,nodes): # pass in the netlist and matrices
    # y_add is the matrix of admittances
    # netlist is the list of lists to analyze
    # currents is the vector of currents
    # voltages is the vector of voltages
    # num_nodes is the number of nodes
    
    x=len(nodes)                                    #Finding the length of nodes and storing in variable x
    
    
    for comp in netlist:                            # for each component...
        #print(' comp ', comp)                       # which one are we handling...

        
        if ( comp[COMP.TYPE] == COMP.R ):           # a resistor
            i = comp[COMP.I] -1
            j = comp[COMP.J] -1
            if (i >= 0):                            # add on the diagonal
                y_add[i,i] += 1.0/comp[COMP.VAL]
            
            if (j>=0):                          
                y_add[j,j] += 1.0/comp[COMP.VAL]
                
            if(i>=0 and j>=0):                      # for other positions to fill in R matrix
                y_add[i,j] += -1.0/comp[COMP.VAL]
                y_add[j,i] += -1.0/comp[COMP.VAL]
            
        if ( comp[COMP.TYPE] == COMP.IS ):          # a current           
            i = comp[COMP.I] -1                     # current logic 
            j = comp[COMP.J] -1
            if(i>=0):
                currents[i] += -comp[COMP.VAL]
            if(j>=0):
                currents[j] += comp[COMP.VAL]
                
                
                
        if ( comp[COMP.TYPE] == COMP.VS ):           # a voltage
            i = comp[COMP.I] -1                      #voltage logic
            j = comp[COMP.J] -1 
            x+=1
            currents[x-1] += comp[COMP.VAL]
            if(i>=0):
                
                y_add[x-1,i] = 1.0
                y_add[i,x-1] = 1.0
            if(j>=0):
                
                y_add[x-1,j] = -1.0
                y_add[j,x-1] = -1.0
                
        
    return y_add,currents,voltages  # need to update with new value

############################################################
# Start the main program now...                            #
############################################################

# Read the netlist!
netlist = read_netlist()

# Print the netlist so we can verify we've read it correctly
for index in range(len(netlist)):
    print(netlist[index])
print("\n")

node,max_node= ranknetlist(netlist)
print(' Nodes ', node, ' total Nodes ', max_node)
y_add= np.zeros([max_node,max_node])

currents=np.zeros([max_node,1])
voltage=np.zeros([max_node,1])

new_y,newi,newv=stamper(y_add,netlist,currents,voltage,max_node,node)

print(new_y) # prints the new R matrix


b=solve(new_y,newi)

print(b) 








