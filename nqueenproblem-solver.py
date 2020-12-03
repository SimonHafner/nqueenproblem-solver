# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:09:57 2020

@author: simon
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random as rand


def generate_board(n):
    return np.zeros((n, n))

#M for matrice
def initial_problem_state(M):
    n = len(M)
    problem = []
    for i in range(0, n):
        t = rand.randint(0,n-1)
        problem.append((t, i))
    return problem

# to transform the matrice to a list of tuples:
def transform_to_array(M, problem):
    M2 = M.copy()
    for i in problem:
        M2[i[0],i[1]] = 1
    return M2

def transform_to_matrice(array):
    M = np.zeros((len(array), len(array)))
    for i in array:
        M[i[0],i[1]] = 1
    return M

# to create a delta state

def delta_state(state):
    delta_state = state.copy()
    x = rand.randint(0, len(state)-1)
    y = rand.randint(0, len(state)-1)
    delta_state[x] = (y,x) 
    return delta_state
    

# movement functions for tuple arguments:

def move_left(t):
    t = (t[0], t[1]-1)
    return t
    
def move_right(t):
    t = (t[0], t[1]+1)
    return t
    
def move_up(t):
    t = (t[0]-1, t[1])
    return t

def move_down(t):
    t = (t[0]+1, t[1])
    return t

def move_up_left(t):
    t = move_up(t)
    t = move_left(t)
    return t

def move_up_right(t):
    t = move_up(t)
    t = move_right(t)
    return t
    
def move_down_left(t):
    t = move_down(t)
    t = move_left(t)
    return t
    
def move_down_right(t):
    t = move_down(t)
    t = move_right(t)
    return t
################################

# to search for threats according to one queen placed on t
def individual_cost(t0, problem_state):
    cost = 0
    #1.1 search up
    t1 = t0
    while t1[0] != 0:
        t1 = move_up(t1)
        if problem_state[t1[0],t1[1]] == 1:
            cost += 1

    #1.2 search down
    t1 = t0
    while t1[0] != len(problem_state)-1:
        t1 = move_down(t1)
        if problem_state[t1[0],t1[1]] == 1:
            cost += 1

    #2.1 search left
    t1 = t0
    while t1[1] != 0:
        t1 = move_left(t1)
        if problem_state[t1[0],t1[1]] == 1:
            cost += 1

    #2.2 search right
    t1 = t0
    while t1[1] != len(problem_state)-1:
        t1 = move_right(t1)
        if problem_state[t1[0],t1[1]] == 1:
            cost += 1

    #3.1 search up-left
    t1 = t0
    for i in range(len(problem_state)):
        if t1[1] == 0:
            break
        elif t1[0] == 0:
            break
        t1 = move_up_left(t1)
        if problem_state[t1[0],t1[1]] == 1:
            cost += 1

    #3.1 search up-right
    t1 = t0
    for i in range(len(problem_state)):
        if t1[0] == 0:
            break
        elif t1[1] == len(problem_state)-1:
            break
        t1 = move_up_right(t1)
        if problem_state[t1[0],t1[1]] == 1:
            cost += 1
    
    #3.1 search down-left
    t1 = t0
    for i in range(len(problem_state)):
        if t1[0] == len(problem_state)-1:
            break
        elif t1[1] == 0:
            break
        t1 = move_down_left(t1)
        if problem_state[t1[0],t1[1]] == 1:
            cost += 1
                
    #3.2 search down-right
    t1 = t0
    for i in range(len(problem_state)):
        if t1[0] == len(problem_state)-1:
            break
        elif t1[1] == len(problem_state)-1:
            break
        t1 = move_down_right(t1)
        if problem_state[t1[0],t1[1]] == 1:
            cost += 1
            
    return cost  

def total_cost(v, problem_state):
    total_cost = 0
    for i in v:
        total_cost += individual_cost(i, problem_state)
    return total_cost


def hill_climbing(problem, n):
    state = problem.copy()
    state_M = transform_to_matrice(state)
    for i in range(0, n): 
        if total_cost(state, state_M) == 0:
            print("Success!")
            return state, state_M
        else:
            initial = delta_state(state)
            initial_M = transform_to_matrice(state)
            if total_cost(state, state_M) - total_cost(initial, initial_M) < 0:
                state = state
            else:
                state = initial.copy()
                state_M = initial_M.copy()
    return state, state_M

#functions for simulated annealing:
    
def calc_T(t, k, lambd):
    return float(k * math.exp((-lambd) * t))

def probability_function(delta_e, T):
    return float(math.exp((-delta_e) / T))

def swap_condition(delta_e, probability, rand_numb):
    if delta_e < 0:
        return 1
    if delta_e == 0 and rand_numb > 0.5:
        return 1
    if delta_e > 0 and probability > rand_numb:
        return 1
    else:
        return 0
    

def simulated_annealing(problem, k, lambd, n):
    state = problem.copy()
    state_M = transform_to_matrice(state)
    for t in range(0,n):
        if total_cost(state, state_M) == 0:
            print("Success!")
            return state, state_M
        else:
            initial = delta_state(state)
            initial_M = transform_to_matrice(state)
            T = calc_T(t, k, lambd)
            delta_e = float(total_cost(initial, initial_M) - total_cost(state, state_M))
            probability = probability_function(delta_e, T)
            rand_numb = float(rand.uniform(0,1))
            if swap_condition(delta_e, probability, rand_numb) == 1:
                state = initial.copy()
                state_M = initial_M.copy()
    return state, state_M
        

# creating a problem set:
    
empty_board = generate_board(6)
problem = problem_state(empty_board)
problem_board = transform_to_array(empty_board, problem)

#finding the solution to the problem:

Solution, Solution_M = hill_climbing(problem, 10000)
total_cost(Solution, Solution_M)


Solution, Solution_M = simulated_annealing(problem, 200, 0.05, 10000)
total_cost(problem, problem_board)
