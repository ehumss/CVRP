#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from gurobipy import *
from collections import namedtuple
from bokeh.plotting import figure
from bokeh.io import output_file, save, show

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return (abs(customer1.x - customer2.x)  + abs(customer1.y - customer2.y))

def solve_it(input_data):

    # parse the input
    lines = input_data.split('\n')
    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))

    clients = customers[1:]

    cust_dist = {}
    for i in customers:
        for j in customers:
            if i.index != j.index:
                cust_dist[i.index, j.index] = length(i,j)

    m = Model("CVRP")

    # Adding variables

    vehicles = {}
    for i in range(vehicle_count-1):
        vehicles[i] = m.addVar(vtype=GRB.BINARY, name="vehicle: " +str(i))

    path = {}
    for i in customers:
        for j in customers:
            if i.index != j.index:
                path[i.index,j.index] = m.addVar(vtype=GRB.BINARY, name="Path   "+ str(i.index) +"  "+ str(j.index))

    vCap = {}
    for i in customers:
        if i.index !=0:
            vCap[i.index] = m.addVar(lb=i.demand, ub=vehicle_capacity, name="vCap   "+str(i.index))

    m.update()

    obj = quicksum(path[i.index,j.index]*cust_dist[i.index, j.index] for i in customers for j in customers if i.index!=j.index)

    m.setObjective(obj)

    for j in clients:
        m.addConstr(quicksum(path[i.index,j.index] for i in customers if i.index != j.index) == 1)
    for i in clients:
        m.addConstr(quicksum(path[i.index, j.index] for j in customers if i.index != j.index) == 1)

    for i in clients:
        m.addConstr(vCap[i.index] <= vehicle_capacity + (i.demand - vehicle_capacity)*path[0,i.index])

    for i in clients:
        for j in clients:
            if i!=j:
                m.addConstr(vCap[i.index] - vCap[j.index] + vehicle_capacity*path[i.index,j.index] <=\
                            vehicle_capacity - j.demand)

    m.optimize()
    f = open("./result/solution.csv",'w')
    print('Variable', file=f)
    for v in m.getVars():
        if v.x != 0:
            print('%s' % v.varName, file=f)

    return round(m.ObjVal,3)

def post_proc(ip_data):

    # Reading the solution file generated
    data = pd.read_csv("./result/solution.csv")

    # Creating a Pandas data frame from the output file
    df = data[data['Variable'].str.startswith('Path')]

    # Creating an array of tuples corresponding between two locations
    routes = []
    for i, row in df.iterrows():
        x = row['Variable'].split()
        routes.append([x[1], x[2]])
    routes_copy = routes[:]

    # Determining the number of vehicles needed
    route_counter = 0
    for i in routes:
        if str(i[0]) == '0':
            route_counter += 1

    vehicles = route_counter

    # Creating a dictionary that holds each of the tour/routes
    tour = {}
    for j in routes_copy:
        if route_counter > 0 and j[0] == '0':
            tour["Route " + str(route_counter)] = j
            route_counter -= 1

    for key, value in tour.items():
        ip_data[key] = ""
        for i in value:
            if value[-1] != '0':
                for j in routes_copy:
                    if value[-1] == j[0]:
                        value.append(j[1])
    df = pd.DataFrame.from_dict(tour, orient='index')
    df = df.T

    for key, value in tour.items():
        for ind, val in df[key].iteritems():
            if val is not None:
                if (int(ind) != 0 and int(val) != 0):
                    ip_data[key].iloc[int(val)] = int(ind)
                else:
                    ip_data[key].iloc[0] = 0
    return vehicles, ip_data

import sys
if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()

        tour_length = solve_it(input_data)

        print("The total tour lenght is : " + str(tour_length) + " units")

        # Reading the input file to a Pandas dataframe for further processing
        ip_data = pd.read_csv(file_location, sep=' ', header=None, skiprows=[0], skip_blank_lines=True,
                              names=['Demand', 'x-cor', 'y-cor']).dropna(axis=0)

        vehicles, op_data = post_proc(ip_data)

        print("The optimal solution is to use " + str(vehicles) + " vehicles for this problem")
        print("")
        print("The path to be followed is shown in the below grid: ")
        print(op_data)

        # Reading output file

    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')







