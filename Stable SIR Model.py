#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekly project 3. Silulate the virus epidemic with SIR Model
Due on: September 20, 11:59 PM

*You are encouraged to delete my hints and start from blank, programmers!*
"""
import numpy as np
import matplotlib.pyplot as plt
#=============================================================================
#Section 1. set the model parameters
#=============================================================================
#1.1 set the total population, let's say there are 3500 people on the island
n = 3500
#1.2 set the initial infected group as 1% of the population (round to integer)
init_infect = 0.01 * n
#1.3 set the initial recovered group as 0% of the population (round to integer)
init_immune = 0
#1.4 set the transmission rate as 0.8
beta = 0.8
#1.5 set the recover rate as 0.05
gamma = 0.05

#=============================================================================
#Section 2. define the initial status table
#=============================================================================
#2.1 generate a 1x3 matrix filled with 0, name is as sir

sir = np.zeros((1,3))

#2.2 replace the first column of the sir matrix with initial susceptible group

sir[0,0] = n

#2.3 replace the second column with initial infected group

sir[0,1] = init_infect

#2.4. replace the third column with initial recovered group

sir[0,2] = init_immune

# up to here, you should have the initial sir matrix .

# print(sir)

#=============================================================================
#Section 3. simulate the epidemic of the virus
#=============================================================================
#3.1 make a copy of the initial status table

sir_sim = sir.copy()

#3.2 create empty lists to keep tracking the changes

susceptible_pop_norm = [n - init_infect - init_immune] # record of susceptible group

infected_pop_norm = [init_infect] # record of infected group

recovered_pop_norm = [init_immune] # record of recovered group

#3.3 Set the time horizon to 100 days

days = 100

total_days = np.linspace(1,days,num=days) # no need to change. done for you.

#3.4 iterate through the 100 days
for day in total_days:
    
    S = susceptible_pop_norm[int(day-1)]
    
    I = infected_pop_norm[int(day-1)]
    
    R = recovered_pop_norm[int(day-1)]
    
    rate_S = (-beta * S/n * I)
    
    rate_I = beta * S/n * I - gamma*I
    
    rate_R = gamma*I

    new_inm = R + rate_R
    
    new_inf = I + rate_I
    
    new_sus = n - new_inf - new_inm
    
    susceptible_pop_norm.append(new_sus)
    
    infected_pop_norm.append(new_inf)
    
    recovered_pop_norm.append(new_inm)


    # normalize the SIR (i.e., calculate % of population),
outcome = [susceptible_pop_norm,infected_pop_norm,recovered_pop_norm]

for i in range(101):
    outcome[0][int(i)]/=n
    outcome[1][int(i)]/=n
    outcome[2][int(i)]/=n
    
outcome[0].pop(0)
outcome[1].pop(0)
outcome[2].pop(0)

#=============================================================================
#Section 4. Run the following code to visualize the simulation outcome.
# As long as you have section 1 - section 3 completed,
# and stored the simulation result in the outcome list,
# the following part is ready to go, no further change needed.
#=============================================================================

# define the plot function
def sir_simulation_plot(outcome,days):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    days = np.linspace(1,days,num=days)
    susceptible = np.array(outcome[0])*100
    infected = np.array(outcome[1])*100
    recovered = np.array(outcome[2])*100
    ax.plot(days,susceptible,label='susceptible',color='y')
    ax.plot(days,infected,label='infected',color='r')
    ax.plot(days,recovered,label='recovered',color='g')
    ax.set_xlabel('Days')
    ax.set_ylabel('Proportion of the population')
    ax.set_title("SIR Model Simulation")
    plt.legend()
    plt.show()
# call the function to plot the outcome
sir_simulation_plot(outcome,days=days)

