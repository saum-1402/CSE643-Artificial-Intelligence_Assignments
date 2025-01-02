# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque


## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # df = df.astype({
    #     'column1': new_type1,
    #     'column2': new_type2,
    #     # Add more columns as needed
    # })
    # print(df_trips.dtypes)
    # print(df_stop_times.dtypes)
    merged_df = pd.merge(df_trips, df_stop_times, on='trip_id')
    # Create trip_id to route_id mapping

    # Map route_id to a list of stops in order of their sequence
    
    # route_to_stops-------------------------------------------------------
    for i in merged_df['route_id'].unique():
        stops = merged_df[merged_df['route_id'] == i]['stop_id'].unique()
        route_to_stops[i] = stops.tolist()
    #-----------------------------------------------------------------------
    # print("route_to_stops_done")


    # trip_to_route--------------------------------------------------------
    # trip_ids = df_trips['trip_id'].unique()
    # for i in trip_ids:
    #     route_ids = df_trips[df_trips['trip_id'] == i]['route_id'].unique()
    #     trip_to_route[i] = route_ids.tolist()
    
    trip_to_route = df_trips.groupby('trip_id')['route_id'].apply(list).to_dict()
    #-----------------------------------------------------------------------
    # print(trip_to_route)
    # print("trips_to_route_done")

    # Count trips per stop
    # stop_trip_count------------------------------------------------------
    stop_ids = df_stop_times['stop_id'].unique().tolist()
    for i in (stop_ids):
        count_of_trips = len(df_stop_times[df_stop_times['stop_id'] == i]['trip_id'])
        stop_trip_count[i] = count_of_trips
    


    # Create fare rules for routes
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id')
    routes = merged_fare_df['route_id'].unique()
    for i in routes:
        fare_rules[i] = merged_fare_df[merged_fare_df['route_id'] == i]
    
    # Merge fare rules and attributes into a single DataFrame

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """

    out = []
    for key,value in route_to_stops.items():
        s=0
        for j in value:
            s+=stop_trip_count[j]
        out.append((key,s))
    out = sorted(out, key=lambda item: item[1],reverse=True)
    return out[:5]
    # pass  # Implementation here

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """

    out = []
    for key,value in stop_trip_count.items():
        out.append((key,value))
    out = sorted(out, key=lambda item: item[1],reverse=True)
    return out[:5]
    pass  # Implementation here

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    d=defaultdict(int)
    for route,stops in route_to_stops.items():
        for stop in stops:
            d[stop]+=1
    
    out = []
    for key,value in d.items():
        out.append((key,value))
    out = sorted(out, key=lambda item: item[1],reverse=True)
    return out[:5]
    pass  # Implementation here

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    out = []
    for route,stops in route_to_stops.items():
        for i in range(len(stops)-1):
            out.append(((stops[i],stops[i+1]),int(route)))
    d=defaultdict(int)
    for i in out:
        d[i]+=stop_trip_count[i[0][0]]+stop_trip_count[i[0][1]]
    out = sorted(out, key=lambda item: item[1],reverse=True)
    return out[:5]
    pass  # Implementation here

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    # merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id')

    return merged_fare_df

# Visualize the stop-route graph interactively
#source: https://networkx.org/documentation/stable/tutorial.html#
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    G = nx.Graph()

    for route, stops in route_to_stops.items():
        G.add_nodes_from(stops)
        for i in range(len(stops) - 1): 
            G.add_edge(stops[i], stops[i + 1])
    
    pos = nx.spring_layout(G, k=0.05, seed=42)

    plt.figure(figsize=(15, 15)) 

    nx.draw_networkx_nodes(
        G, pos, node_size=20, node_color='skyblue', edgecolors='black', linewidths=0.5, alpha=0.7
    )

    nx.draw_networkx_edges(G, pos, width=0.3, edge_color='gray', alpha=0.5)

    plt.savefig('graph_repr.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    pass  # Implementation here

# import time
# import psutil

def do():
    pass
#     large_data = [x for x in range(10000000)] 


# def memory_usage():
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     mem_used_mb = mem_info.rss / 1024 / 1024
#     return mem_used_mb


# def get_analysis(start_time,end_time, after_run_mem, before_run_mem,func,step_count):
#     print(f'Memory used for {func}: {after_run_mem - before_run_mem} MiB')
#     print(f'Time taken for {func}: {end_time-start_time} seconds')
#     print(f'Steps taken for {func}: {step_count}')
    # return end_time-start_time,after_run_mem-before


# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    # before_run_mem = memory_usage()
    # start_time = time.time()
    global route_to_stops

    do() 
    output = []
    step_count = 0
    #in this i need to ensure that 
    for key,value in route_to_stops.items():
        step_count+=1
        if start_stop in value and end_stop in value:
            output.append(key)  # add route_id
        else:
            step_count+=1
            continue #dont add
    # end_time = time.time()
    # after_run_mem = memory_usage()
    # get_analysis(start_time,end_time,after_run_mem,before_run_mem,"direct_route_brute_force",step_count)
    return output

    # pass  # Implementation here

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms

    
    # OptimalRoute(X, Y, R) <= DirectRoute(X, Y) & (R == X)

    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print


    # Define Datalog predicates
    
    # DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y)
    DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y)


    # OptimalRoute(R, X, Y, Z) <= (DirectRoute(R, X, Z) & DirectRoute(R, Z, Y) & (X != Z) & (Z != Y) & (X != Y)) or (DirectRoute(R1, X, Z) & DirectRoute(R2, Z, Y) & (R1 != R2) & (X != Z) & (Z != Y) & (X != Y) & OptimalRoute(R1, R2, X, Y, Z))
 

    OptimalRoute(R1, R2, X, Y, Z) <= RouteHasStop(R1, X) & RouteHasStop(R2, Y) & RouteHasStop(R1, Z) & RouteHasStop(R2, Z) & (X != Y)

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # print(len(route_to_stops))
    for route,values in route_to_stops.items():
        for stop in values:
            + RouteHasStop(int(route),int(stop))

    pass  # Implementation here

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    # before_run_mem = memory_usage()
    # start_time = time.time()
    step_count = 1
    do()
    t = DirectRoute(R, start, end).data
    # print(t)
    step_count+=1
    out = []
    for i in t:
        step_count+=1
        out.append(i[0])
    # print(t)
    # print(type(t))
    # end_time = time.time()
    # after_run_mem = memory_usage()
    # get_analysis(start_time,end_time,after_run_mem,before_run_mem,"query_direct_routes",step_count)
    step_count+=1
    return out[::-1]
    # return t.sort()

    # pass  # Implementation here

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """


    # OptimalRoute(R1, R2, X, Y, Z) <= RouteHasStop(R1, X) & RouteHasStop(R2, Y) & RouteHasStop(R1, Z) & RouteHasStop(R2, Z) & (X != Y) & (X != Z) & (Y != Z)
    step_count=0
    # before_run_mem = memory_usage()
    # start_time = time.time()
    step_count += 1
    do()
    t = OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include).data
    # print(t)
    step_count += 1
    out = []

    if(t):
        for i in t:
            step_count += 1
            l = []
            step_count += 1
            l.append(i[0])
            l.append(stop_id_to_include)
            l.append(i[1])
            out.append(tuple(l))

    # end_time = time.time()
    # after_run_mem = memory_usage()
    # get_analysis(start_time,end_time,after_run_mem,before_run_mem,"forward_chaining",step_count)
    return out
    pass  # Implementation here

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """

    step_count = 0

    # before_run_mem = memory_usage()
    # start_time = time.time()
    step_count += 1
    do()
    t = OptimalRoute(R1, R2, end_stop_id, start_stop_id, stop_id_to_include).data
    # print(t)
    out = []
    if(t):
        for i in t:
            step_count += 1
            l = []
            l.append(i[0])
            l.append(stop_id_to_include)
            l.append(i[1])
            out.append(tuple(l))

    # end_time = time.time()
    # after_run_mem = memory_usage()
    # get_analysis(start_time,end_time,after_run_mem,before_run_mem,"backward_chaining",step_count)
    return out
    pass  # Implementation here

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    pass  # Implementation here

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here
