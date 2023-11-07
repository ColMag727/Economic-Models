# Magie Zheng
"""
Project 5. In this project, you will apply Object Oriented Programming to
simulate the market demand/supply model.

"""
import numpy as np
import matplotlib.pyplot as plt
import random


np.random.seed(380)
#=============================================================================
# Section 1. Define classes
#=============================================================================
# 1.1. define a class Econ_agent
# attributes: id_number, budget
# methods: introduce_me(self):
# print out agent's id number and budget in sentences.


class Econ_agent:

    def __init__(self, id_number, budget):
        self.id_number = id_number
        self.budget = budget

    def introduce_me(self):
        print(self.id_number + self.budget)


# 1.2 define  "Consumer" as a child class of Econ_agent;
# 1.2.1 the Consumer class inherits all attributes and methods from Econ_agent;
# 1.2.2 additional attributes:
    # (1) 'preference', indicates how much the consumer like the product
    # 0 means does not like at all, 1 means wiling to spend whole budget on it

    # (2). wtp (i.e.,willingness to pay).
    # the wtp is the product of budget and preference

# 1.2.3 additional methods:
    # buying(self,price)
    # the buying method defines the consumer's purchasing decision
    # it takes two parameter, the first one is self, the second one is 'price'
    # when wtp is smaller than price, the consumer won't buy;
    # when wtp is greater or equal to price, the consumer will buy the product.
    # depending on budget, each customer may purchase no more than 5 units.
    # the buying method returns if the consumer decide to buy.


class Consumer(Econ_agent):

    def __init__(self, id_number, budget, preference):
        super().__init__(id_number, budget)
        self.preference = preference
        self.wtp = self.budget * self.preference

    def buying(self, price):
        if self.wtp >= price and self.budget > 0:
            max_units = min(5, int(self.budget / price))
            return max_units
        else:
            return 0
    
    


# 1.3 define "Producer" as a child class of Econ_agent;
# 1.3.1 the Producer class inherits all attributes and methods from Econ_agent;
# 1.3.2 additional attributes:
    # 1. 'opp_cost', indicate the opp_cost of producing and selling one unit
    # assume opp_cost is a constant for each producer.
# 1.3.3. additional methods:
    # selling(self,price)
    # the selling method defines the producer's decision
    # it takes two parameter, the first one is self, the second one is 'price'
    #when opp_cost is greater than price, the producer won't produce
    # when opp_cost or equal to price, the consumer will produce
    # the quantity supplied depends on the ratio between the producer's budget
    # and the the opp_cost. i.e., quantity supplied = budget/opp_cost
    # the method returns total amount a producer made.


class Producer(Econ_agent):
    
    def __init__(self, id_number, budget, opp_cost):
        super().__init__(id_number, budget)
        self.opp_cost = opp_cost
        
    def selling(self,price):
        quantity = 0
        
        if self.opp_cost <= price:
            
            quantity = self.budget / self.opp_cost
            
        return quantity
    
    def get_Id(self):
        return self.id_number
    
    def get_budget(self):
        return self.budget
    
    def get_opp_cost(self):
        return self.opp_cost
    
# consumer1 = Consumer(1, 500, 0.7)
# producer1 = Producer(101, 3000, 150)

# consumer1.introduce_me()
# print(f"Consumer's WTP: ${consumer1.wtp:.2f}")

# units_to_buy = consumer1.buying(40)
# print(f"Consumer wants to buy {units_to_buy} units.")

# producer1.introduce_me()
# revenue = producer1.selling(60)
# print(f"Producer's revenue: ${revenue:.2f}")           

#=============================================================================
# Section2. generate objects
#=============================================================================
# 2.1 generate a list of 200 consumer
# each has a unique id number
# budget is determined by a random draw from a normal distribution
#   set the normal distribution mean = 500, s.d. = 100
# preference is determined by a random draw from a uniform distribution [0,1]
consumer_list = []
for i in range(1, 201):
    budget = max(0, np.random.normal(500, 100))
    preference = random.uniform(0, 1)
    consumer_list.append(Consumer(i, budget, preference))



#2.2 generate a list of 50 producers
# each has a unique id number
# budget is determined by a random draw from a UNIFORM distribution [1000,8000]
# opp_cost is determined by  a drandom draw from uniform distribution [100,200]
producer_list = []
for i in range(1, 51):
    budget = random.uniform(1000, 2000)
    opp_cost = random.uniform(100, 200)
    producer_list.append(Producer(i, budget, opp_cost))
    
    


#=============================================================================
# Section 3. Simulate the market mechanism, and find the equilibrium
#=============================================================================

# Specifically, find the market equilibrium price and quantity (i.e., you
# may not be able to find a point where  Q = S exactly. Let's consider the
# market is in equilibrium as long as the difference between D and S is smaller
# than 5 units)

def calculate_total_quantity_demanded(consumers, price):
    return sum(consumer.buying(price) for consumer in consumers)

def calculate_total_quantity_supplied(producers, price):
    return sum(producer.selling(price) for producer in producers)

equilibrium_price = 100
equilibrium_quantity = 110

for price in np.linspace(100,200,400):
    # print(price)
    total_demand = calculate_total_quantity_demanded(consumer_list, price)
    total_supply = calculate_total_quantity_supplied(producer_list, price)

    if abs(total_demand - total_supply) <= 5:
        equilibrium_price = price
        equilibrium_quantity = min(total_demand, total_supply)
        break

print(f"Market Equilibrium Price: {equilibrium_price}")
print(f"Market Equilibrium Quantity: {equilibrium_quantity}")

#=============================================================================
# Section4. Define the demand curve and supply curve
#=============================================================================
# Define the price range from 100 to 200
price_range = np.linspace(100,200,400)
demand_curve = [calculate_total_quantity_demanded(consumer_list, price) for price in price_range]
supply_curve = [calculate_total_quantity_supplied(producer_list, price) for price in price_range]

# 4.3 visualize the demand and supply, see if it makes sense.

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(supply_curve, price_range, label='Supply Curve', color='red')
plt.plot(demand_curve, price_range, label='Demand Curve', color='blue')


# Labeling and titles
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Demand and Supply Curves')
plt.grid(True)

# Add legend
plt.legend()

# Show the plot
plt.show()


#=============================================================================
# Section 5. Changes in supply
#=============================================================================
# imagine there is a technology improvement, reduce the average opp_cost by 5%
# run a simulation to find the new market equilibrium
# visualize the change graphically

reduction_percentage = 5
reduced_opp_costs = [producer.opp_cost * (1 - reduction_percentage / 100) for producer in producer_list]
newProducers = []
for i in range(len(producer_list)):
    producer = Producer(producer_list[i].get_Id(), producer_list[i].get_budget(), reduced_opp_costs[i])
    newProducers.append(producer)
    
    
new_equilibrium_price = equilibrium_price
new_equilibrium_quantity = equilibrium_quantity

# sometimes the differences may larger than 5 all the time when the price adding by one
for price in np.linspace(100, 200, 400):
    total_supply = calculate_total_quantity_supplied(newProducers, price)
    total_demand = calculate_total_quantity_demanded(consumer_list, price)
    
    # print(f"S: {total_supply}")
    # print(f"D: {total_demand}")
    if abs(total_demand - total_supply) <= 5:
        new_equilibrium_price = price
        new_equilibrium_quantity = min(total_demand, total_supply)
        break

print(f"New Market Equilibrium Price (5% Reduction): {new_equilibrium_price}")
print(f"New Market Equilibrium Quantity (5% Reduction): {new_equilibrium_quantity}")


plt.figure(figsize=(10, 6))
plt.plot(supply_curve, price_range, label='Supply Curve(Original)', color='green')
plt.plot([calculate_total_quantity_supplied(newProducers, price) for price in price_range] , price_range, label='Supply Curve (Reduced)', color='orange')
plt.plot(demand_curve, price_range, label='Demand Curve', color='blue')
plt.xlabel('Quantity')
plt.ylabel('Price')
plt.title('Market Supply Curve Change (5% Reduction)')
plt.grid(True)
plt.legend()
plt.show()

#=============================================================================
# Section 6 (optional). Estimate the demand and supply function
# Estimate the demand function and supply function you simulated in Section 4,
# and then use the two function to figure out the market equilibrium,
# see if the calculated equilibrium coincide with the simulation result.
#=============================================================================
