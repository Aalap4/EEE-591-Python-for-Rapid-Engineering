




#Aalap Paragbhai Doshi
#ASU ID 1217130629



# Retirement Savings calculator
import numpy as np
from tkinter import *                                             # import everything from tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

fields_GUI = ('Mean Return(in %)', 'Std Dev Return(in %)', 'Yearly Contribution(in $)',
          'No. of Years of Contribution', 'No. of Years to Retirement','Annual Spend in Retirement') # Fields in the GUI Interface

years_needed = 70                                           #the number of years the analysis requires
N = 10                                                      # to calculate the savings N times and average

def cal(entries) :
        for iter in range(N) : 
            yoc = (int(entries['No. of Years of Contribution'].get()))         # number of contrbibuting years
            yor = (int(entries['No. of Years to Retirement'].get()))           # number of years to retirement
            y_c = (float(entries['Yearly Contribution(in $)'].get()))          #annual contribution
            y_r = (float(entries['Annual Spend in Retirement'].get()))         #yearly retirement spend
            mean = (float(entries['Mean Return(in %)'].get()))
            sigma = (float(entries['Std Dev Return(in %)'].get()))
        
            noise = (sigma*0.01)*np.random.randn(years_needed)                  #random noise
            
            years = np.arange(0,70,1)
            last_amt = []                                    #array to store the remaining amount for each of the 10 analyses
            wealth= np.zeros(len(years))                     # wealth earned annually
            for year in years[:-1] :
                wealth[year+1]= wealth[year]*(1+(mean/100)+noise[year])         #wealth at each year
                if (year <= yoc ) :
                    wealth[year+1] += y_c
                if (year > yor) :
                    wealth[year+1] -= y_r
            wealth = np.where(wealth>0,wealth,None)                             #stopping when wealth goes below $0
            
            last_amt.append(wealth[yor])        #adding the remaining wealth at each of the 10 analyses to an array
                
            plt.plot(years,wealth,marker='x') #plotting
        plt.xlabel('Years')
        plt.ylabel('Wealth Remaining($)')
        shw = Label(create, text = np.average(last_amt), anchor=W) #displaying the average retirement wealth in the GUI
        shw.grid(row=7, column=1, sticky=W)
        plt.show() 
            

def makeform(master, fields_GUI):
   dict = {}                                              # create an empty dictionary
   for i in range(len(fields_GUI)):                       # for each of the fields to create
      lab = Label(master, width=22, text=fields_GUI[i]+": ", anchor=E)
      ent = Entry(master)
      ent.insert(0,"0")

      lab.grid(row=i, column=0, sticky=E, padx=5, pady=5)
      ent.grid(row=i, column=1, sticky=E, padx=5, pady=5)

      dict[fields_GUI[i]] = ent                           # add it to the dictionary

   return dict                                            # and return the dictionary

# start the main program
create = Tk()                                             # create a GUI
make_fields = makeform(create, fields_GUI)                       # make the fields

create.configure(background='Black')

calculation = Label(create, text='Retirement Wealth (in $) : ')
calculation.grid(row=7, column=0, sticky=E, padx=5)

button1 = Button(create, text='CALCULATE', command=(lambda e=make_fields: cal(e)), fg="black", font=('times',11,'bold'))
button1.grid(row=8, column=0, padx=5, pady=5, sticky=S)            

button2 = Button(create, text='Quit', command=create.destroy, fg="red", font=('times',11,'bold'))
button2.grid(row=8, column=1, padx=5, pady=5, sticky=S)          


create.mainloop()                              # start execution






