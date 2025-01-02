import numpy as np
import pickle
from queue import Queue
import heapq
import psutil
import os
import time
import json

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]


def pq(adj_matrix,node):
    pq = []
    for i in range(len(adj_matrix[node])):
        if(adj_matrix[node][i]>0):
            heapq.heappush(pq,(adj_matrix[node][i],i))
    return pq

def dfs(node,adj_matrix,visit,goal_node,path,level,p_level):
  if(p_level>level):
    return False
  
  visit[node]=1
  path.append(node)

  if node == goal_node:
    return True

  priority_queue = pq(adj_matrix,node)
  for i in range(len(priority_queue)):
    n = priority_queue[i][1]
    if(visit[n]==0):
        # print(priority_queue[i])
        if(dfs(n,adj_matrix,visit,goal_node,path,level,p_level+1)==True):
            return True
  # for i in range(len(adj_matrix[node])):
  #   if(adj_matrix[node][i]>0 and visit[i]==0):
  #       # path.append(i)
  #       if(dfs(i,adj_matrix,visit,goal_node,path,level,p_level+1)==True):
  #           return True
  path.pop()
  visit[node]=0   #using this guarantees that the path is the optimal path but since it is time consuming i am not using this
  return False
    
def get_ids_path(adj_matrix, start_node, goal_node):
  level = len(adj_matrix)
  # ans=[]
  # low = 0
  # high = level-1
  # while(low<=high):
  #     mid = (low+high)//2
  #     # print(mid)
  #     path = []
  #     visit = [0]*level
  #     if(dfs(start_node,adj_matrix,visit,goal_node,path,mid,0)==True):
  #         ans = path
  #         high = mid-1
  #     else:
  #         low = mid+1
  # if(len(ans)>0):
  #     return ans
  # return None
      
  for i in range(level):
    path = []
    visit = [0]*level
    if(dfs(start_node,adj_matrix,visit,goal_node,path,i,0)==True):
      return path
    # print(i)
  return None


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def path(parent,target):
    # Reconstruct the path from start to target using parent array
    ans = []
    while target != -1:
        ans.append(target)
        target = parent[target]
    ans.reverse()
    # print(ans)
    return ans

def get_bidirectional_search_path(adj_matrix,start_node,end_node):
    if(start_node==end_node):
        return [start_node]
    n = len(adj_matrix)
    mtr = adj_matrix
    q_start = Queue()
    q_target = Queue()

    q_start.put(start_node)
    q_target.put(end_node)

    parent_st = [-1]*len(mtr)
    parent_t = [-1]*len(mtr)

    visit_st = [0]*len(mtr)
    visit_st[start_node]=1

    visit_t = [0]*len(mtr)
    visit_t[end_node]=1

    while (not q_start.empty()) and (not q_target.empty()):
        # if(not q_start.empty()):
        s = q_start.get()
    
        if(visit_t[s]==1):
            l1 = path(parent_st,s)
            l2 = path(parent_t,s)
            l2.reverse()
            return l1+l2[1:]
        
        priority_queue_st = pq(mtr,s)
        # for i in range(n):
        for n in range(len(priority_queue_st)):
            i = priority_queue_st[n][1]
            if(visit_st[i]==0):
                q_start.put(i)
                visit_st[i]=1
                parent_st[i]=s

        t = q_target.get()
        # print(t)
        if(visit_st[t]==1):
            # print(f'{t} inside')
            # print(f'{t} t is')
            l1 = path(parent_st,t)
            # print(parent_st)
            l2 = path(parent_t,t)
            l2.reverse()
            # print(l1,l2)
            return l1+l2[1:]
        
        priority_queue_t = pq(mtr,t)

        
        # for i in range(n):
        for n in range(len(priority_queue_t)):
            i = priority_queue_t[n][1]
            if(visit_t[i]==0):
                q_target.put(i)
                visit_t[i]=1
                parent_t[i]=t
                # print(f'{i} i is')
    return None




# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]


def heuristic(node_attributes, node, start_node, end_node):
    x1 = node_attributes[node]['x']
    y1 = node_attributes[node]['y']

    xs = node_attributes[start_node]['x']
    ys = node_attributes[start_node]['y']

    xe = node_attributes[end_node]['x']
    ye = node_attributes[end_node]['y']

    return ((x1-xs)**2 + (y1-ys)**2)**0.5 + ((x1-xe)**2 + (y1-ye)**2)**0.5

def get_astar_search_path(adj_matrix,node_attributes,start_node,end_node):
    q=[]
    s = heuristic(node_attributes,start_node,start_node,end_node)
    heapq.heappush(q,(s,start_node))
    parent = [-1]*len(adj_matrix)
    visit = [0]*len(adj_matrix)
    # visit[start_node]=1
    INT_MAX = float('inf')
    g_cost = [INT_MAX]*len(adj_matrix)
    g_cost[start_node]=0
    while q:
        p = heapq.heappop(q)
        f_s = p[0]
        s = p[1]
        
        if(s==end_node):
            return path(parent,end_node)

        if(visit[s]==1):
            continue

        visit[s]=1
        
        for i in range(len(adj_matrix[s])):
            if(adj_matrix[s][i]>0 and visit[i]==0):
                if(g_cost[s]+adj_matrix[s][i]<g_cost[i]):
                    g_cost[i]=g_cost[s]+adj_matrix[s][i]
                    g_s = g_cost[i]
                    h_s = heuristic(node_attributes,i,start_node,end_node)
                    f_s = g_s+h_s
                    heapq.heappush(q,(f_s,i))
                    parent[i]=s
        # visit[s]=2
    return None


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node):
    q_st=[]
    q_t=[]

    s_st = heuristic(node_attributes,start_node,start_node,end_node)
    s_t = heuristic(node_attributes,end_node,start_node,end_node)

    heapq.heappush(q_st,(s_st,start_node))
    heapq.heappush(q_t,(s_t,end_node))

    parent_st = [-1]*len(adj_matrix)
    parent_t = [-1]*len(adj_matrix)

    visit_st = [0]*len(adj_matrix)
    visit_t = [0]*len(adj_matrix)
    # visit[start_node]=1
    INT_MAX = float('inf')
    g_cost_st = [INT_MAX]*len(adj_matrix)
    g_cost_st[start_node]=0

    g_cost_t = [INT_MAX]*len(adj_matrix)
    g_cost_t[end_node]=0

    while q_st and q_t:
        p_s = heapq.heappop(q_st)
        p_t = heapq.heappop(q_t)
        f_n_s = p_s[0]
        f_n_t = p_t[0]
        s = p_s[1]
        t = p_t[1]
    
        if(visit_t[s]==1):
            l1 = path(parent_st,s)
            l2 = path(parent_t,s)
            l2.reverse()
            return l1+l2[1:]
        
        visit_st[s]=1
    
        for i in range(len(adj_matrix[s])):
            if(adj_matrix[s][i]>0 and visit_st[i]==0):
                if(g_cost_st[s]+adj_matrix[s][i]<g_cost_st[i]):
                    g_cost_st[i]=g_cost_st[s]+adj_matrix[s][i]
                    g_s = g_cost_st[i]
                    h_s = heuristic(node_attributes,i,start_node,end_node)
                    f_n_s = g_s+h_s
                    heapq.heappush(q_st,(f_n_s,i))
                    parent_st[i]=s

        if(visit_st[t]==1):
            l1 = path(parent_st,t)
            l2 = path(parent_t,t)
            l2.reverse()
            return l1+l2[1:]
            # return path(parent_st,t)+path(parent_t,end_node)
        
        visit_t[t]=1

        for i in range(len(adj_matrix[t])):
            if(adj_matrix[t][i]>0 and visit_t[i]==0):
                if(g_cost_t[t]+adj_matrix[t][i]<g_cost_t[i]):
                    g_cost_t[i]=g_cost_t[t]+adj_matrix[t][i]
                    g_t = g_cost_t[i]
                    h_t = heuristic(node_attributes,i,start_node,end_node)
                    f_n_t = g_t+h_t
                    heapq.heappush(q_t,(f_n_t,i))
                    parent_t[i]=t
        # visit[s]=2
    return None
    

# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

# reference: https://www.youtube.com/watch?v=qrAub5z8FeA : G-55. Bridges in Graph - Using Tarjan's Algorithm of time in and low time: Striver

c=1

def bonus_dfs(node,parent,visit,adj_matrix,time,low,roads):
    global c
    visit[node]=1
    time[node]=c
    low[node]=c
    c+=1
    for i in range(len(adj_matrix[node])):
        if(adj_matrix[node][i]==0 and adj_matrix[i][node]==0):
            continue
        if(i==parent):
            continue
        if(visit[i]==0):
            bonus_dfs(i,node,visit,adj_matrix,time,low,roads)
            low[node]=min(low[node],low[i])
            if(low[i]>time[node]):
                roads.append([node,i])
        else:
            low[node]=min(low[node],time[i])
    return roads

def bonus_problem(adj_matrix):
    n = len(adj_matrix)
    visit = [0]*n
    time = [0]*n
    low = [0]*n
    roads = []
    for i in range(n):
        if(visit[i]==0):
            bonus_dfs(i,-1,visit,adj_matrix,time,low,roads)
    if(len(roads)>0):
        return roads

    return None

def memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_used_mb = mem_info.rss / 1024 / 1024
    return mem_used_mb
    print("Memory Used: {:.2f} MB".format(mem_used_mb))


def cost(output):
  cost = 0
  for j in output:
    if(j==None):
      continue
    for i in range(len(j)-1):
        cost += adj_matrix[j[i]][j[i+1]]
  return cost

def calculate_time1(algo,adj_matrix):
  before_run_mem = memory_usage()
  start_time = time.time()
  pair_path = []
  final = []
  for i in range(0,125):
      for j in range(0,125):
          # d={}
          if i!=j: # since path from a node to itself is just the node I am ignoring it
              if(get_bidirectional_search_path(adj_matrix,i,j)==None):
                  print(f'No path exists between {i} and {j}')
                  continue
              print(f'Pair: {i} and {j}')
              path = algo(adj_matrix,i,j)
              # print(d['memory'])
              # d['path'] = path
              final.append(path)
  # print('Iterative Deepening Search: ')
  end_time = time.time()
  after_run_mem = memory_usage()
  print(f'Memory used: {after_run_mem - before_run_mem} MiB')
  print(f'Time taken: {end_time-start_time} seconds')
  print(f'Cost: {cost(final)}')
  return final




def calculate_time2(algo,adj_matrix,node_attributes):
  before_run_mem = memory_usage()
  start_time = time.time()
  pair_path = []
  final = []
  for i in range(0,125):
      for j in range(0,125):
          if i!=j: # since path from a node to itself is just the node I am ignoring it
              if(get_bidirectional_search_path(adj_matrix,i,j)==None):
                  # print(f'No path exists between {i} and {j}')
                  continue
              # print(f'Pair: {i} and {j}')
              path = algo(adj_matrix,node_attributes,i,j)
              final.append(path)
  end_time = time.time()
  after_run_mem = memory_usage()
  # print(final)
  print(f'Memory used: {after_run_mem - before_run_mem} MiB')
  print(f'Time taken: {end_time-start_time} seconds')
  print(f'Cost: {cost(final)}')
  return final

# def write_output(output, filename):
#   with open(filename, 'w') as f:
#     json.dump(output, f, indent=4)

def write_output(output, filename):
  #write in a text file
  with open(filename, 'w') as f:
    for i in output:
      f.write(str(i))
      f.write('\n')


if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')

  # print('Iterative Deepening Search: ')
  # output1 = calculate_time1(get_ids_path,adj_matrix)
  # write_output(output, 'output_IDS.json')

  # print('Bidirectional Search: ')
  # output2 = calculate_time1(get_bidirectional_search_path,adj_matrix)
  # # write_output(output, 'output_bidirectional_search.json')

  # print('A* Search: ')
  # output3 = calculate_time2(get_astar_search_path,adj_matrix,node_attributes)
  # # write_output(output, 'output_astar_search.json')

  # print('Bidirectional Heuristic Search: ')
  # output4 = calculate_time2(get_bidirectional_heuristic_search_path,adj_matrix,node_attributes)
  # # write_output(output, 'output_bidirectional_heuristic_search.json')

  # print(output4)
  # write_output(output4, 'output_bidirectional_heuristic_search.txt')