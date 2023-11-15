# Vehicle Routing Problem with Capacity Constraint (VRP)
# Python 3.10
import math
import time
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


    ## Set of functions ##
# get_data() -> parse data from file
# nearest_neighbor() -> find neighbor
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

def distance(node1, node2):
    coords_1 = [node1[0],node1[1]]
    coords_2 = [node2[0],node2[1]]
    return math.dist(coords_1, coords_2)


def nearest_neighbor(graph, capacity, depot):
    nodes = list(graph.nodes())
    nodes.remove(depot)

    route = [depot]
    current_capacity = 0

    while nodes:
        nearest_node = min(nodes, key=lambda node: distance(graph.nodes[route[-1]]['pos'], graph.nodes[node]['pos']))

        if current_capacity + graph.nodes[nearest_node]['demand'] <= capacity:
            route.append(nearest_node)
            current_capacity += graph.nodes[nearest_node]['demand']
        else:
            route.append(depot)
            route = route + [depot]
            current_capacity = 0

        nodes.remove(nearest_node)

    return route


def two_opt(route, graph):
    best_route = route
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue  # changes nothing, skip then
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if cost(new_route, graph) < cost(best_route, graph):
                    best_route = new_route
                    improved = True
        route = best_route

    return best_route


def relocate(route, graph):
    best_route = route
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(1, len(route) - 1):
                if i == j or i == j - 1:
                    continue  # no relocation, skip then
                new_route = route[:i] + route[i+1:j] + [route[i]] + route[j:]
                if cost(new_route, graph) < cost(best_route, graph):
                    best_route = new_route
                    improved = True
        route = best_route

    return best_route


def bat_algorithm(graph, capacity, depot, num_bats, max_iterations, alpha, gamma, f_min, f_max, A):
    bats = [nearest_neighbor(graph, capacity, depot) for _ in range(num_bats)]
    f_values = [cost(bat, graph) for bat in bats]

    for _ in range(max_iterations):
        for i in range(num_bats):
            f_i = f_min + (f_max - f_min) * random.random()
            f_i = f_i if random.random() > A else f_i * (1 - np.exp(-gamma * _))
            v_i = [random.uniform(-1, 1) for _ in range(len(bats[i]) - 1)]

            new_bat = relocate(bats[i], graph)

            if random.random() < alpha and cost(new_bat, graph) < f_values[i]:
                bats[i] = new_bat
                f_values[i] = cost(new_bat, graph)

        best_bat_index = np.argmin(f_values)
        best_bat = bats[best_bat_index]

        bats[best_bat_index] = two_opt(best_bat, graph)
        f_values[best_bat_index] = cost(bats[best_bat_index], graph)

    return bats[np.argmin(f_values)]


def draw_scheme(graph, routes):
    pos = nx.get_node_attributes(graph, 'pos')

    plt.figure(figsize=(10, 8))

    # Plot nodes
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=700)

    # Plot edges
    for route in routes:
        route_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=route_edges, edge_color='blue', width=2)

    # Add labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black', font_weight='bold')

    # Add arrows to indicate direction
    for route in routes:
        for i in range(len(route) - 1):
            edge = (route[i], route[i + 1])
            arrow_length = 0.1  # Adjust arrow length as needed
            arrow_props = dict(facecolor='red', edgecolor='red', arrowstyle='->', shrinkA=0, lw=1, alpha=0.7)
            edge_pos = np.array([pos[route[i]], pos[route[i + 1]]])
            arrow_pos = edge_pos[1] - arrow_length * (edge_pos[1] - edge_pos[0])
            plt.annotate("", arrow_pos, xytext=edge_pos[1], arrowprops=arrow_props)

    plt.title("VRPCC Solution")
    plt.axis('off')
    plt.show()

def cost(route, graph):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance(graph.nodes[route[i]]['pos'], graph.nodes[route[i + 1]]['pos'])
    return total_distance


def run(data, vehicle_capacity):
    num_nodes = len(data)
    depot_node = 0

    G = nx.complete_graph(num_nodes)
    pos = {i: (float(data[i][0]), float(data[i][1])) for i in range(num_nodes)}
    demands = {i: int(data[i][2]) for i in range(num_nodes)}

    nx.set_node_attributes(G, pos, 'pos')
    nx.set_node_attributes(G, demands, 'demand')

    # Construction Heuristic (Nearest Neighbor)
    initial_solution = nearest_neighbor(G, vehicle_capacity, depot_node)

    # Local Search: 2-opt
    improved_solution_2opt = two_opt(initial_solution, G)

    # Local Search: 1-0 Relocate
    final_solution = relocate(improved_solution_2opt, G)

    # Solve VRPCC using Bat Algorithm
    bat_solution = bat_algorithm(G, vehicle_capacity, depot_node, num_bats=10, max_iterations=10, alpha=0.2, gamma=0.1, f_min=0.1, f_max=0.7, A=0.9)

    # Visualize the solutions
    draw_scheme(G, [final_solution, bat_solution])


# This is our main function implementing our functionality
def main():
    # Starting timer
    start_time = time.time()

    run(get_data("data.txt"), vehicle_capacity=160)

    # Stopped timer
    print("\nExecution Time: "+datetime.fromtimestamp(time.time() - start_time).strftime('%S:%M:%H') + " seconds\n")
    return




# Run script
main()