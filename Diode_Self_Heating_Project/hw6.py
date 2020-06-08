################################################################################
# HW6 Diode Temperature                                                                                                     #
################################################################################

import numpy as np                  # import required packages
from scipy import optimize
import matplotlib.pyplot as plt

################################################################################
# Define required constants                                                                                              #
################################################################################

MIL = 25               # conversion factor
UM2CM = 1e-4           # square centimeters in meters
LAYERS = 5            # number of layers

# Thermal Coefficients W/cmC - these are the layers we need to solve!
#   0  Si active       10 mil  1.3
#   1  Si bulk         15 mil  1.3
#   2  Cu back metal   5 mil   3.86
#   3  paste           25 mil  0.5
#   4  Metal flag      100 mil 5
K = np.array([1.3,1.3,3.86,0.5,5.0],float)

# thickness of each layer in cm
THICK = np.array([10,15,5,25,100],float)*MIL*UM2CM
NUM_LAYERS = len(THICK) #Number of Layers in diode
# Diode Constants
I0 = 3e-9                   # reverse bias saturation current
Q = 1.6e-19                 # charge on the electron
KB = 1.38e-23               # Boltzmann constant
IDE = 2.0                   # Ideality factor
AREA = 10*UM2CM*10*UM2CM;   # Area of the diode

MAX_SOURCE_V = 10           # Maximum source voltage to analyze
MIN_SOURCE_V =  1           # Minimum source voltage to analyze
V_STEP = 0.1                # Voltage step size
R1 = 1e3                    # values of the various resistors
R2 = 4e3
R3 = 3e3
R4 = 2e3
R5 = 1e3
TBASE = 325                 # Constant temperature of the base

################################################################################
# Solve for pseudo 2D self heating based on stack. Start at the base layer     #
# which is at a constant temperature. Then work our way up.                    #
# Inputs:                                                                      #
#    current -    current through the structure                                #
#    voltage -    voltage across the structure                                 #
#    tbase   -    temperatue of the bottom layer                               #
# Outputs:                                                                     #
#    temp    -    temperature of the top layer                                 #
################################################################################

def calc_temp(current,voltage,tbase):
    Power = current*voltage # equation for heat flow
    H_flow = Power #heat flow is the power
    T_res = THICK/(K*AREA) #calulate the thermal resistance
    temp=tbase
    for layer in range(NUM_LAYERS):
        temp = temp + H_flow*T_res[layer]
    return(temp)

################################################################################
# Return diode current and temperature error                                   #
# Inputs:                                                                      #
#    volt     - voltage across the device                                      #
#    tguess   - guess at device temperature                                    #
#    tbase    - temperature of the bottom layer                                #
# Outputs:                                                                     #
#    curr     - current through the diode                                      #
#    t_err    - difference between the guess and the calculated value          #
################################################################################

def diode_i_terr(volt,tguess,tbase):
    #Diode Current calculation
    thresh_v = IDE * KB * tbase / Q
    d_curr = I0 * (np.exp(volt/thresh_v)-1)
    #temperature of diode at topmost layer
    t_calc = calc_temp(d_curr,volt,tbase)
    return( d_curr, t_calc-tguess )

################################################################################
# Bridge with a diode with node and temperature error. Based on the guess of   #
# voltages, compute diode current and temp error. Then compute the current at  #
# each node, which should sum to 0. Return these sums and the temp error.      #
# Inputs:                                                                      #
#    parmams - an array containing the following values fsolve will optimize:  #
#              index 0 - the voltage at node 1                                 #
#              index 1 - the voltage at node 2                                 #
#              index 2 - the voltage at node 3                                 #
#              index 3 - the temperature of the diode                          #
#    source_v - the source voltage                                             #
#    tbase    - the temperate of the base layer                                #
# Outputs:                                                                     #
#    An array of the sum of the currents at each node plus the temperature err #
################################################################################

def f(params,source_v,tbase):
    n1_v   = params[0]       # extract the parameters
    n2_v   = params[1]
    n3_v   = params[2]
    tguess = params[3]

    # compute the diode current and temperature error
    diode_c, diode_t_err = diode_i_terr((n1_v-n2_v),tguess,tbase)
    # based on the calculated current and received voltages, sum the currents
    # at each node
    n1_c = (n3_v-n1_v)/R1 - diode_c - n1_v/R2
    n2_c = (n3_v-n2_v)/R3 + diode_c - n2_v/R4
    n3_c = (source_v-n3_v)/R5 - (n3_v-n1_v)/R1 - (n3_v-n2_v)/R3
    
    return([n1_c,n2_c,n3_c,diode_t_err])  # goal is all values to be 0!


################################################################################
# This is the main loop. We create arrays to hold the data for each value of   #
# the voltage source. Then we execute the loop for each source voltage.        #
################################################################################

source_vals = np.arange(MIN_SOURCE_V,MAX_SOURCE_V,V_STEP) # voltages to use
n1_vals = np.zeros_like(source_vals)              # node 1 voltages
n2_vals = np.zeros_like(source_vals)              # node 2 voltages
n3_vals = np.zeros_like(source_vals)              # node 3 voltages
t_vals  = np.zeros_like(source_vals)              # temperature values
i_vals  = np.zeros_like(source_vals)              # diode current values

temp = TBASE          # initial guesses
v1 = MIN_SOURCE_V     # after first value, we use the last calculated as the
v2 = MIN_SOURCE_V     # initial values
v3 = MAX_SOURCE_V 

for index in range(len(source_vals)):
    [v1,v2,v3,temp] = optimize.fsolve(f,[v1,v2,v3,temp],(source_vals[index],TBASE))
    n1_vals[index] = v1
    n2_vals[index] = v2
    n3_vals[index] = v3
    t_vals[index] = temp
    i_vals[index] = diode_i_terr((v1-v2),temp,TBASE)[0]
# Generate various plots
plt.figure()
ax = plt.subplot(111)
plt.plot(source_vals,n1_vals,color='red',label='Node 1')
plt.plot(source_vals,n2_vals,color='blue',label='Node 2')
plt.plot(source_vals,n3_vals,color='green',label='Node 3')
box = ax.get_position()
#legend box
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='lower center', bbox_to_anchor=(0.8,0), shadow=False, ncol=1)

plt.xlabel('Source Voltage')
plt.ylabel('Node Voltages')
plt.title("Source Voltage vs Node Voltage")


# Plot of Source Voltage vs  Diode temperature
plt.figure()
plt.plot(source_vals,t_vals)
plt.xlabel('Source Voltage')
plt.ylabel('Diode Temperature')
plt.title("Source Voltage vs  Diode temperature")


# Plot of Source Voltage vs log(Diode Current)
plt.figure()
plt.plot(source_vals,np.log(i_vals))
plt.xlabel('Source Voltage')
plt.ylabel('log(Diode Current)')
plt.title(" Source Voltage vs log(Diode Current)")
plt.show()
