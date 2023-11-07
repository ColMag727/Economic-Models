# Magie Zheng

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project 6. optimization and consumer theory

In this project, you will be practicing on root finding and optimization problems
In addition, you will apply the computational method to solve the utility
maximization problem in economics.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import minimize
import scipy.optimize as opt

#=============================================================================
# Section 1. Root finding and optimization
#=============================================================================
# 1.1. define the function y = ln(x) + (x-6)^3 - 4x + 30
# you can find the printed equation on Canvas
def f(x):
    return np.log(x) + (x - 6) ** 3 - 4 * x + 30

# 1.2. plot the function on the domain [1, 12]
x = np.linspace(1, 12, 100)
y = f(x)

def plot_function(x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='y = ln(x) + (x-6)^3 - 4x + 30', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of y = ln(x) + (x-6)^3 - 4x + 30')
    plt.grid(True)
    plt.legend()
    plt.show()
    
plot_function(x,y)

# 1.3. derive and define the first-order derivative of the function
def fp(x):
    return 1/x + 3*(x-6)**2 - 4


# 1.4. plot it on the domain [1, 12]

plot_function(x, fp(x))

# 1.5. Define the Newton-Raphson algorithm (as a function)
def newton_raphson(f,fp, initial_guess, tolerance=1e-9 , max_iteration=100):
    """
    This function will apply the Newton-Raphson method to find the root of a given function
    Parameters:
    ------------
    f : function
       The original function we wanted to find roots
    fp : function
        The first order derivative of the original function
    initial_guess: list
        A list of starting points.
    tolerance : float, optional
        defines how close to zero needs to be
    max_iteration : int
        defines maximum iterations if not converge
    Return:
    roots : list
        a list of roots found
    """
    roots = [ ]
    for guess in initial_guess:
        # x = guess
        # for i in range(max_iteration):
        #     delta_x = f(x) / fp(x)
        #     x -= delta_x
        #     if abs(x) < tolerance:
        #         roots.append(x)
        #         break
        x = guess
        for i in range(max_iteration):
            x_next = x - f(x) / fp(x)
            if x_next<0:
                x_next = 0.1
            if abs(x-x_next) < tolerance:
                roots.append(x)
                break
            x = x_next


    return roots

def fpp(x):
    return -x**-2 + 6*(x-6)

# 1.6.Use the Newton-Raphson algorithm you defined to find the root of the function
# store the result in a list named as res_1
initial_guess = [7]
res_1 = newton_raphson(f,fp,initial_guess)



# 1.7. use the Newton-Raphson method to find the
# maximum value on the domain [4, 8], name the returned variable as res_2
re = newton_raphson(fp, fpp, [4,7])
f(re[0])
f(re[1])
res_2 = re[0]

# 1.8. use the Newton-Raphson method to find the
# minimum value on the domain [4, 8], name the returned variable as res_3
res_3 = re[1]






def neg_f(x):
    return -(np.log(x) + (x - 6)**3 - 4*x + 30)
# 1.9. use the scipy.optimize library to
# (a). find the root of f(x), store the result in variable res_4
result_a = root(f, x0=3)
# print(result_a)
# Store the result in res_4
res_4 = result_a.x[0]

print("Root of f(x):", res_4)


# (b). find miniumn value of f(x) on the domain [4, 8],
# name the returned var as res_5
domain = [(4, 8)]
result_b = minimize(f, x0=6, bounds=domain, method='L-BFGS-B')

# Extract the minimum value
res_5 = result_b.fun

print("Minimum value of f(x) on [4, 8]:", res_5)

# (3). find maximum value of f(x) on the domain [4, 8],
# name the returned var as res_6
result_c = minimize(neg_f, x0=6, bounds=domain, method='L-BFGS-B')

# Extract the maximum value (which is the negation of the minimum)
res_6 = -result_c.fun

print("Maximum value of f(x) on [4, 8]:", res_6)



#=============================================================================
# Section 2. Utility Theory and the Application of Optimization
#=============================================================================

# Consider a utility function over bundles of A (apple) and B (banana)
#  U(B, A) =( B^alpha) * (A^(1-alpha))
# hint: you can find the printed equation on Canvas: project 7.

# 2.1. Define the given utility function
def utility(A, B, alpha):
    return (B**alpha) * (A**(1 - alpha))

# 2.2. Set the parameter alpha = 1/3,
# Assume the consumer always consume 1.5 units of B.
# plot the relationship between A (x-axis) and total utility (y-axis)
# set the range of A between 1 and 10
A_values = np.linspace(1, 10, 100)
alpha = 1/3
B = 1.5
total_utility = [utility(A,B,alpha) for A in A_values]


def plot_utility(A, u_level):
    plt.figure(figsize=(10, 6))
    plt.plot(A_values, total_utility, label=f'Utility (alpha={alpha})', color='blue')
    plt.xlabel('A (Apples)')
    plt.ylabel('Total Utility')
    plt.title('Total Utility vs. A (Apples)')
    plt.grid(True)
    plt.legend()
plt.show(plot_utility(A_values,total_utility))


# 2.3.  plot the 3-dimensional utility function
# 3-d view of utility
A = np.linspace(0.1, 10, 100)
B = np.linspace(0.1, 2, 100)
B,A = np.meshgrid(B, A)
u_level = utility(A, B, 1/3)


def plot_utility_3d(A, B, u_level):
    fig = plt.figure(figsize=(12,12))
    ax_3d = fig.add_subplot(1,1,1,projection="3d")
    ax_3d.contour3D(A,B,u_level,20,cmap=plt.cm.Blues)
    ax_3d.set_xlabel("A (Apples)")
    ax_3d.set_ylabel("B (Bananas)")
    ax_3d.set_zlabel("Total Utility")

plot_utility_3d(A, B, u_level)



# 2.4.plot the utility curve on a "flatten view"
A = np.linspace(1, 10, 100)
B = np.linspace(1, 10 ,100).reshape((100,1))
u_level = utility(A, B, 1/3)

def plot_utility_flat(A, B, u_level):
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    contours = ax1.contourf(A, B.flatten(), u_level, cmap=plt.cm.Blues)
    fig.colorbar(contours)
    ax1.set_xlabel("Consumption of X1")
    ax1.set_ylabel("Consumption of X2")
    ax1.set_title("Indifference map")
    plt.show()

plot_utility_flat(A, B, u_level)

# 2.5. from the given utitlity function, derive A as a function of B, alpha, and U
# plot the indifferences curves for u=1 ,3,5,7,9 on the same figure.
# Put B on the x-axis, and A on the y-axis


def A_indifference(B, ubar, alpha=1/3):
    ln_b = np.log(B)
    ln_u = np.log(ubar)

    A = np.exp((ln_u - alpha * ln_b) / (1 - alpha))

    return A


fig3 = plt.figure(figsize=(8, 8))
ax3 = fig3.add_subplot(1, 1, 1)
fig4 = plt.figure(figsize=(8, 8))
ax4 = fig4.add_subplot(1, 1, 1)

def plot_indifference_curves(ax, alpha=1/3):
    for u in [1, 3, 5, 7, 9]:
        b = np.linspace(0.5, 20, 1000)
        a = A_indifference(b, u)
        ax4.plot(b, a, label=f"u={u}")


plot_indifference_curves(ax3)
plot_indifference_curves(ax4)



# # Define a range of utility levels
# U_values = [1, 3, 5, 7, 9]
# # Create a figure
# plt.figure(figsize=(10, 6))
# # Plot indifference curves for each utility level
# for U in U_values:
#     A_values = ((U / (B_values**alpha))**(1 / (1-alpha)))
#     plt.plot(A_values, B_values, label=f'U={U}')
# # Set labels and title
# plt.xlabel('B')
# plt.ylabel('A')
# plt.title('Indifference Curves (alpha=1/3)')
# # Add a legend
# plt.legend()
# # Show the plot
# plt.grid(True)
# plt.show()

# 2.6.suppose pa = 2, pb = 1, Income W = 20,
# Add the budget constraint to the  previous figure

def plot_budget_constraint(pa, pb, w, ax):

    b = np.linspace(0, w / pb, 1000)
    a = (w - b * pb) / pa
    ax.plot(b, a, label="budget constraint", color="black")
    ax.fill_between(b, a, alpha=0.3)


plot_budget_constraint(2, 1, 20, ax3)
plot_budget_constraint(2, 2, 20, ax4)
ax3.set_xlabel("B")
ax3.set_ylabel("A")
ax3.set_title("Indifference curves and budget constraint for pa=2, pb=1")
ax3.legend()
ax4.set_xlabel("B")
ax4.set_ylabel("A")
ax4.set_title("Indifference curves and budget constraint for pa=2, pb=2")
ax4.legend()
plt.show()

# 2.7. find the optimized consumption bundle and maximized utility
def objective(pa, pb, w, alpha=1 / 3):

    def utility_fn(x):
        return -utility(x[0], x[1], alpha)

    def budget_fn(x):
        return w - pa * x[0] - pb * x[1]

    x0 = np.array([w / pa, 0])
    res = opt.minimize(utility_fn, x0, constraints={"type": "eq", "fun": budget_fn})
    return res


opt1 = objective(2, 1, 20)
res7 = {
    "optimal_B": opt1.x[1],
    "optimal_A": opt1.x[0],
    "optimal_utility": -opt1.fun
}
opt2 = objective(2, 2, 20)
res8 = {
    "optimal_B": opt2.x[1],
    "optimal_A": opt2.x[0],
    "optimal_utility": -opt2.fun
}

print(f"When pa=2, pb=1, and w=20, the result is {res7}")
print(f"When pa=2, pb=2, and w=20, the result is {res8}")