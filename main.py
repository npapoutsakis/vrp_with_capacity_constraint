# Vehicle Routing Problem with Capacity Constraint (VRP)
# Python 3.10

import time
from datetime import datetime
from builtins import type

    ## Set of functions ##
# get_data() -> parse data from file
# distance() -> distance between nodes
# construct_route() -> list data
# two_opt() -> heuristic
# relocate() -> heuristic
# bat_algorithm() -> main algorithm, provides best path
# draw_scheme() -> plot graph

def get_data(filepath):
    # open file for reading
    f = open(filepath, 'r')

    # empty list to add coordinates of nodes/customers
    data = []

    # get the num of nodes (customers)
    total_nodes = int(f.readline())

    # get capacity value of vehicle
    capacity = int(f.readline())

    # gather all coordinates
    for i in range(2*total_nodes):
        if i < total_nodes:
            node_info = f.readline().split()
            data.append([node_info[1], node_info[2]])
        else:
            demand = f.readline().split()
            data[i%total_nodes].append(demand[1])
    # each element in data list has a tuple like (x-coord, y-coord, demand)
    return data



def construct_route():
    return
def two_opt():
    return
def relocate():
    return
def bat_algorithm():
    return
def draw_scheme():
    return


# This is our main function implementing our functionality
def main():
    # Starting timer
    start_time = time.time()


    get_data("data.txt")


    # Stopped timer
    print("\nExecution Time: "+datetime.fromtimestamp(time.time() - start_time).strftime('%S:%M:%H') + " seconds\n")
    return


# Run script
main()