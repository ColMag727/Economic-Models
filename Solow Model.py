# Magie Zheng
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm
"""
    This class will create an instance of the Solow-Swan growth model

    arguments:
    ------------------
    para_dict: dict
        a dictionary of parameters
    state_dict: dict
        a dictionary of model state

    attributes
    --------------

    methods:
    -------------

"""
class Growth_Model:
    def __init__(self,para_dict=None, state_dict=None):

        # read-in the given parameters and variables
        self.para_dict = para_dict
        self.state_dict = state_dict

        # calculate lower case y
        self.state_dict['y'] = (self.para_dict['a']
                                * self.state_dict['k']**self.para_dict['alpha'])
        # calculate upper case K (i.e., aggregate capital)
        self.state_dict['K'] = self.state_dict['k'] * self.state_dict['L']
        self.state_dict['Y'] = self.state_dict['y'] * self.state_dict['L']
        self.state_dict['d'] = self.state_dict['k'] * (self.para_dict['delta']
                                                        + self.para_dict['n'])
        self.state_dict['i'] = self.state_dict['y'] * self.para_dict['s']
        # Calculate upper case I (aggregate investment)
        self.state_dict['I'] = self.state_dict['i'] * self.state_dict['L']

        self.steady_state = {}

        self.init_param = copy.deepcopy(para_dict)
        self.init_state = copy.deepcopy(state_dict)

    def get_param(self):
        return self.para_dict

    def get_state(self):
        return self.state_dict
    
    def get_steady_state(self):
        return f"""
        s = {self.para_dict['s']}:
        k_star = {self.steady_state['k_star']}
        y_star = {self.steady_state['y_star']}
        c_star = {self.steady_state['c_star']}
        i_star = {self.steady_state['i_star']}
        
        """

        
    def check_model(self):
        # Print the attributes of the instance
        attributes = vars(self)
        for attr in attributes:
            print(f"{attr}: {attributes[attr]}")
 
    def __load_param(self):
        self.para_dict = copy.deepcopy(self.init_param)
        return self.para_dict.get('n')[0], self.para_dict.get('s')[0], self.para_dict.get('alpha')[0], \
            self.para_dict.get('delta')[0], self.para_dict.get('a')[0]    
            

    def find_steady_state(self):
        # step 1. load parameters
        n, s, alpha, delta, a = self.__load_param()

        # step 2. find the steady state
        k_t = np.linspace(1, 35, 40000000)  # create the k_t domain

        tolerance = 1e-7  # set the tolerance

        num_iter = 0
        for k in tqdm(k_t, desc=f'Finding steady state when s = {self.para_dict["s"]}', unit='iteration'):
            # calculate the steady state
            y = a * k ** alpha
            i = s * y
            d = (delta + n) * k
            num_iter += 1
            if abs(i - d) < tolerance:
                self.steady_state['k_star'] = k
                self.steady_state['y_star'] = y
                self.steady_state['c_star'] = y - i
                self.steady_state['i_star'] = i
                break
        self.num_iter = num_iter
        return
    
    def growth(self, years):
        # reset to initial status
        time_line = np.linspace(0, years, num=years + 1, dtype=int)
        # return to the initial state
        self.state_dict = copy.deepcopy(self.init_state)
        n, s, alpha, delta, a = self.__load_param()
        # step 2. examine growth
        for t in time_line:
            # 2.2. load all current states
            y_t = self.state_dict.get('y')
            k_t = self.state_dict.get('k')
            Y_t = self.state_dict.get('Y')
            L_t = self.state_dict.get('L')
            K_t = self.state_dict.get('K')
            i_t = self.state_dict.get('i')
            I_t = self.state_dict.get('I')
            d_t = self.state_dict.get('d')

            # 2.3 calculate new states i.e., the dynamic
            dk = s * y_t[t] - (delta + n) * k_t[
                t]  # change in capital per worker, formula is delta_k = s*y_t - (delta+n) * k_t
            k_next = k_t[t] + dk  # capital per worker in the next period
            L_next = L_t[t] * (1 + n)  # population in the next period
            d_next = (delta + n) * k_next  # break-even investment per worker in the next period
            y_next = a * k_next ** alpha  # income per worker in the next period
            K_next = k_next * L_next  # aggregate capital in the next period
            Y_next = y_next * L_next  # aggregate income in the next period
            i_next = s * y_next  # investment per worker in the next period
            I_next = i_next * L_next  # aggregate investment in the next period

            # 2.4. update the state_dict
            k_t = np.append(k_t, k_next)
            y_t = np.append(y_t, y_next)
            Y_t = np.append(Y_t, Y_next)
            K_t = np.append(K_t, K_next)
            L_t = np.append(L_t, L_next)
            i_t = np.append(i_t, i_next)
            I_t = np.append(I_t, I_next)
            d_t = np.append(d_t, d_next)

            # update the attributes
            self.state_dict['k'] = k_t
            self.state_dict['y'] = y_t
            self.state_dict['Y'] = Y_t
            self.state_dict['K'] = K_t
            self.state_dict['L'] = L_t
            self.state_dict['i'] = i_t
            self.state_dict['I'] = I_t
            self.state_dict['d'] = d_t

    def plot_growth(self):
        # create a figure
        fig = plt.figure(figsize=(10, 6))
        # add a subplot
        plot_ax = fig.add_subplot(111)
        # plot income per worker against time
        time_line = np.linspace(0, len(self.state_dict['y']) - 1, num=len(self.state_dict['y']), dtype=int)
        plot_ax.plot(time_line, self.state_dict['y'], label='income per worker')
        # set the title
        plot_ax.set_title('Income per worker against time')
        # set x label
        plot_ax.set_xlabel('year')
        # set y label
        plot_ax.set_ylabel('income per worker')
        # add legend
        plot_ax.legend()
        # show the plot
        plt.show()
        return

    def plot_income_growth(self):
        # create a figure
        fig = plt.figure(figsize=(10, 6))
        # add a subplot
        plot_ax = fig.add_subplot(111)
        # plot income per worker
        plot_ax.plot(self.state_dict['k'], self.state_dict['y'], label='income per worker')
        # plot investment per worker
        plot_ax.plot(self.state_dict['k'], self.state_dict['i'], label='investment per worker')
        # plot break-even investment per worker
        plot_ax.plot(self.state_dict['k'], self.state_dict['d'], label='break-even investment per worker')
        # set the title
        plot_ax.set_title(
            'Income per worker, investment per worker, and break-even investment against capital per worker')
        # set x label
        plot_ax.set_xlabel('capital per worker')
        # set y label
        plot_ax.set_ylabel('Output per worker')
        # add legend
        plot_ax.legend()
        # show the plot
        plt.show()
        return
    


if __name__ == '__main__':
    # Section 3. Specify model parameters and examine economic growth
    # set parameters (extraneously given):
    parameters = {'n': np.array([0.002]),  # pop growth rate
                  's': np.array([0.15]),  # saving rate
                  'alpha': np.array([1 / 3]),  # share of capital
                  'delta': np.array([0.05]),  # depreciation rate
                  'a': np.array([1])  # factor productivity
                  }

    states = {'k': np.array([0.1]),  # default capital per worker
              'L': np.array([100]),  # default population
              }

    # instantiate a growth model
    model = Growth_Model(para_dict=parameters, state_dict=states)
    # simulate the growth
    model.growth(200)
    # 3.2  visualize the growth
    model.plot_growth()
    # visualize the income per worker against time
    model.plot_income_growth()
    # 3.2  find the steady state of the model
    model.find_steady_state()
    print(model.get_steady_state())

    # Section 4. Use the growth model class to perform "what-if" analysis.
    # 4-1 set the saving rate to 0.33
    parameters['s'] = np.array([0.33])
    # instantiate a growth model
    model = Growth_Model(para_dict=parameters, state_dict=states)
    # simulate the growth
    model.growth(200)
    # plot the growth
    model.plot_growth()
    # visualize the income per worker against time
    model.plot_income_growth()
    # find the steady state
    model.find_steady_state()
    print(model.get_steady_state())

    # 4-2 set the saving rate to 0.5
    parameters['s'] = np.array([0.5])
    # instantiate a growth model
    model = Growth_Model(para_dict=parameters, state_dict=states)
    # simulate the growth
    model.growth(200)
    # plot the growth
    model.plot_growth()
    # visualize the income per worker against time
    model.plot_income_growth()
    # find the steady state
    model.find_steady_state()
    print(model.get_steady_state())


