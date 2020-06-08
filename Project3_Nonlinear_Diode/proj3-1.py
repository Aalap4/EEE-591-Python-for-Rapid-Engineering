# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:09:26 2020

@author: Aalap
"""
#Project3
# Importing the necessary libraries

import numpy as np                    
import matplotlib.pyplot as plt                                                
from scipy import optimize

import warnings
warnings.filterwarnings("ignore")   # turn off iteration warnings
#defining the constants

#ALL GLOBAL VARIABLES AND CONSTANTS
TEMP=375                                                 #CONSTANT      # temperature for problem 2
AREA=1e-8                                                #CONSTANT      # temperature for problem 2

Is =1e-9                                                                #Subthreshold current  
T = 350                                                  #CONSTANT      #temperature for problem 1
K = 1.38064852e-23                                       #CONSTANT      #Boltzmann constant
Q = 1.60217662e-19                                       #CONSTANT      # Charge of electron
r =10000                                                 #VARIABLE      #resistor value

phi_value=0.8                                                           # phi for problem 2                                                               
ideality_factor=1.5                                                     #ideality factor for Part2                                                          

P1_VDD_STEP=0.1                                                         #initial guess of diode voltage

ID = []
diode_v = []                                                            #list to store diode voltages for PART1
Vs = []                                                                 #list to store voltage soures values for PART1
tol = 1e-8                                                              #tolerance value
N = 100                                                 #CONSTANT       #no.of iterations
error=0                                                                 #initial value of error
i=0                                                                   
iD =[]                                                                  #list
V =[]                                                                   #list for PART2
I=[]                                                                    #list for PART2

r1=11000                                               #VARIABLE        #resistor value for PART1
n=1.7                                                                   #ideality factor(n) for Part1

#################################################

#Part 1

#################################################

# Function to compute diode current
def compute_diode_current(Vd,ide,temp1,subthresh_current):
    Vt = ide*(K*temp1)/Q                                                        #calculate thermal voltage
    Id = subthresh_current*(np.exp(Vd/(Vt))-1)                                  #diode current formula
    return Id                                      

def diode_voltage(Vd,Vs,R_l,ide,temp1,subthresh_current):
    Id = compute_diode_current(Vd,ide,temp1,subthresh_current)                  #diode current formula
    error = ((Vd-Vs)/R_l)+Id                                                    # find error  
    return error                                     

Vd = 0.1                                                   
source_voltage = np.arange(0.1,2.5,0.1)                
                          
for vs1 in source_voltage:
    Vd_guess = Vd                                                              
    Vd = optimize.fsolve(diode_voltage,Vd_guess,(vs1,r1,n,T,Is),maxfev=0,factor=100,xtol=1e-12) #optimize value of diode voltage
    Id = compute_diode_current(Vd,n,T,Is)             #find diode current for the corresponding value
    diode_v.append(Vd)                                #append
    Vs.append(vs1)
    ID.append(Id)

fig, ax1 = plt.subplots()
ax1.plot(Vs,np.log10(ID), 'r')
ax1.set_xlabel("Voltage (in V)")
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Diode Current vs Source Voltage (log scale)', color='r')
ax1.tick_params('y', colors='r')

ax2 = ax1.twinx()
ax2.plot(diode_v,np.log10(ID), 'b')
ax2.set_ylabel('Diode Current vs Diode Voltage (log scale)', color='b')
ax2.tick_params('y', colors='b')

fig.tight_layout()
plt.show()

#########################################

# Start of Problem 2

#########################################

VoltageSource=np.zeros(51)
Measured_DiodeCurrent=np.zeros(51)
f=open('DiodeIV.txt','r')          
lines=f.readlines()
index=0
for j in lines:
    VoltageSource[index]=j.split(' ')[0]
    Measured_DiodeCurrent[index]=j.split(' ')[1]
    index=index+1
del_factor=1e-15

def opt_r(r_value,ide_value,phi_value,area1,temp1,src_v,measured_i):
    estimate_v   = np.zeros_like(src_v)                   # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)                        # an array to hold the diode currents
    prev_v = P1_VDD_STEP                                  # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area1 * temp1 * temp1 * np.exp(-phi_value * Q / ( K * temp1 ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(diode_voltage,prev_v,
				(src_v[index],r_value,ide_value,temp1,is_value),
                                xtol=1e-6)[0]
        estimate_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(estimate_v,ide_value,temp1,is_value)
    return measured_i - diode_i

def opt_phi(phi_value,ide_value,r_value,area1,temp1,src_v,measured_i):
    estimate_v   = np.zeros_like(src_v)                  # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)                       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area1 * temp1 * temp1 * np.exp(-phi_value * Q / ( K * temp1 ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(diode_voltage,prev_v,
				(src_v[index],r_value,ide_value,temp1,is_value),
                                xtol=1e-6)[0]
        estimate_v[index] = prev_v                      # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(estimate_v,ide_value,temp1,is_value)
    return (measured_i - diode_i)/(measured_i + diode_i + del_factor)

def opt_idel(ide_value,phi_value,r_value,area1,temp1,src_v,measured_i):
    estimate_v   = np.zeros_like(src_v)                 # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)                      # an array to hold the diode currents
    prev_v = P1_VDD_STEP                                # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area1 * temp1 * temp1 * np.exp(-phi_value * Q / ( K * temp1 ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(diode_voltage,prev_v,
				(src_v[index],r_value,ide_value,temp1,is_value),
                                xtol=1e-6)[0]
        estimate_v[index] = prev_v                     # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(estimate_v,ide_value,temp1,is_value)
    return (measured_i - diode_i)/(measured_i + diode_i + del_factor)

for i in range(N) :
    
    r_val_opt = optimize.leastsq(opt_r,r,
                                 args=(ideality_factor,phi_value,AREA,TEMP,
                                       VoltageSource,Measured_DiodeCurrent))
    r = r_val_opt[0][0]
    
    phi_val_opt = optimize.leastsq(opt_phi,phi_value,
                                 args=(ideality_factor,r,AREA,TEMP,
                                       VoltageSource,Measured_DiodeCurrent))
    phi_value=phi_val_opt[0][0]
    
    idel_val_opt = optimize.leastsq(opt_idel,ideality_factor,
                                 args=(phi_value,r,AREA,TEMP,
                                       VoltageSource,Measured_DiodeCurrent))
    ideality_factor=idel_val_opt[0][0]
    
    print("\n -----------------------------------\n")
    print("\n Iterations= ", i)
    
    
    error_cal = opt_idel(ideality_factor,phi_value,r,AREA,TEMP,VoltageSource,Measured_DiodeCurrent)
    error= sum(abs(error_cal))/len(error_cal)
    if error< tol :
        break
    else :
        print(error)
    
    print("\n resistance (R)= ",r,"\n barrier_height (Phi)= ",phi_value,"\n Ideality_Factor (n)= ",ideality_factor)
    

newI=AREA * TEMP * TEMP * np.exp(-phi_value * Q / ( K * TEMP ) )

source_voltage1 = np.arange(0,5.1,0.1)     
Vdd=0.1 
           
for vs2 in source_voltage1 :
    Vd_guess = Vdd                                                              
    Vdd = optimize.fsolve(diode_voltage,Vd_guess,(vs2,r,ideality_factor,TEMP,newI),xtol=1e-6) # optimizing diode voltage for given range of source voltages
    Id_2 = compute_diode_current(Vdd,ideality_factor,TEMP,newI)               #compute diode current for corresponding diode voltage          
    V.append(vs2)
    I.append(Id_2)


fig, ax3 = plt.subplots()
ax3.plot(VoltageSource,np.log10(Measured_DiodeCurrent), 'r')
ax3.set_xlabel("Source Voltage")
# Make the y-axis label, ticks and tick labels match the line color.
ax3.set_ylabel('Measured Diode Current (log scale)', color='r')
ax3.tick_params('y', colors='r')

ax4 = ax3.twinx()
ax4.plot(source_voltage1,np.log10(I), 'b')
ax4.set_ylabel('Model Diode Current (log scale)', color='b')
ax4.tick_params('y', colors='b')

fig.tight_layout()
plt.show()