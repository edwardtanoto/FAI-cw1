#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 23:19:44 2021

@author: qianzhang
"""

import sys
from collections import deque

from utils import *
#%matplotlib inline
import matplotlib.pyplot as plt
import random
import heapq
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations



# ______________________________________________________________________________
# Abstract class of problem and Node
# expand to generate successors; 
# path_actions and path_states to recover aspects of the path from the node.  

class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When you create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""
    
    # **kwds Keywords arguments, a dictionary type 
    
    def __init__(self, initial=None, goal=None, **kwds):  #here self defining the problem itself
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        raise NotImplementedError #return possible actions given state s, it regarded as sucessor functions in the slides
    def result(self, state, action): raise NotImplementedError # return result state given by start state and valid action
    def is_goal(self, state):        return state == self.goal
    def action_cost(self, s, a, s1): return 1 #travel from state s to s1 cost 1, s=>start state, a=>action, s1=>resulting state,
    def h(self, node):               return 0
    
    def __str__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.initial, self.goal) #define canonical representation of problem. Problemname(initial state, goal)   
    
class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0): #here self defining the current node object
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __repr__(self): return '<{}>'.format(self.state) #print out format of node, <state name>
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent)) #depth of node
    def __lt__(self, other): return self.path_cost < other.path_cost #define the way of comparing two nodes.
    
    
failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution. state = 'failure'
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off. state = 'cutoff'
    
    
def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s): #problem.actions(s) return all the possible actions on state s
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost) #(yield)can be viewed as generator, expand node will yield several nodes (neighbours/frontier) reachable from the current node
        
def path_actions(node):
    "The sequence of actions to get to this node." #solution
    if node.parent is None:
        return []  
    return path_actions(node.parent) + [node.action] #list of actions from current node to root node

def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None): 
        return []
    return path_states(node.parent) + [node.state] #list of states from current node to root node


# ______________________________________________________________________________
# Queue implementation
# First-in-first-out and Last-in-first-out queues, and a `PriorityQueue`, which allows you to keep a collection of items, and continually remove from it the item with minimum `f(item)` score.

FIFOQueue = deque # check with help(deque), add to the right most, remove from the left most

LIFOQueue = list #add to the right most, remove the right most

class PriorityQueue: #used in best first search
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x): 
        self.key = key
        self.items = [] # a heap of (score, item) pairs
        for item in items:
            self.add(item)
         
    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item) #pair is a tuple type
        heapq.heappush(self.items, pair) #put the node to the sorted priority queue at the correct location w.r.t. value of key(item)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]
    
    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)


# ______________________________________________________________________________
# Search Algorithms
# Best-first search with various f(n) functions gives us different search algorithms. 
# Note that A*, and greedy search can be given a heuristic function, h, but if h is not supplied they use the problem's default h function (if the problem does not define one, it is taken as h(n) = 0).
    
def breadth_first_tree_search(problem): #edit by qz
    "Search shallowest nodes in the search tree first."
    node = Node(problem.initial)
    frontier = FIFOQueue([node]) # queue
    while frontier: #frontier is not empty
        node = frontier.pop() #remove and return the right most element => front
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            frontier.appendleft(child) # add to the left most => tail
    return failure

def depth_first_tree_search(problem):#edit by qz
    "Search deepest nodes in the search tree first."
    node = Node(problem.initial)
    frontier = LIFOQueue([node])
    while frontier:
        node = frontier.pop() #remove the right most element
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            frontier.append(child)
    return failure

def depth_limited_tree_search(problem, limit=10): #edit by qz
    "Search deepest nodes in the search tree first."
    frontier = LIFOQueue([Node(problem.initial)])
    result = failure
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        elif len(node) >= limit:
            result = cutoff
        elif not is_cycle(node):
            for child in expand(problem, node):
                frontier.append(child)
    return result

def best_first_tree_search(problem, f):
    "A version of best_first_search without the `reached` table."
    frontier = PriorityQueue([Node(problem.initial)], key=f)
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            if not is_cycle(child):
                frontier.add(child)
    return failure

def g(n): return n.path_cost    

def is_cycle(node, k=30):
    "Does this node form a cycle of length k or less?"
    def find_cycle(ancestor, k):
        return (ancestor is not None and k > 0 and
                (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1)))
    return find_cycle(node.parent, k)

def uniform_cost_search(problem):
    "Search nodes with minimum path cost first."
    return best_first_tree_search(problem, f=g)

def astar_tree_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n), with no `reached` table."""
    h = h or problem.h # if h=None, h = problem.h, otherwise, h=h
    return best_first_tree_search(problem, f=lambda n: g(n) + h(n)) #n is input, g(n)+h(n) is returned value of lambda function f 

def breadth_first_bfs(problem):
    "Search shallowest nodes in the frontier search tree first; using best-first."
    #this is a best first tree search implementation 
    return best_first_tree_search(problem, f=len)

def depth_first_bfs(problem):
    #this is a best first tree search implementation 
    "Search deepest nodes in the frontier search tree first; using best-first."
    return best_first_tree_search(problem, f=lambda n: -len(n))

# ______________________________________________________________________________
# Python code in FAI_Search.ipynb
# romania

class Map:
    """A map of places in a 2D world: a graph with vertexes and links between them. 
    In `Map(links, locations)`, `links` can be either [(v1, v2)...] pairs, 
    or a {(v1, v2): distance...} dict. Optional `locations` can be {v1: (x, y)} 
    If `directed=False` then for every (v1, v2) link, we add a (v2, v1) link."""

    def __init__(self, links, locations=None, directed=False):
        if not hasattr(links, 'items'): # Distances are 1 by default
            links = {link: 1 for link in links}
        if not directed:
            for (v1, v2) in list(links):
                links[v2, v1] = links[v1, v2]
        self.distances = links
        self.neighbors = multimap(links)
        self.locations = locations or defaultdict(lambda: (0, 0)) #set (0,0) as the default location if argument locations is None

        
def multimap(pairs) -> dict: # '-> dict' means returned value is a dictionary type
    "Given (key, val) pairs, make a dict of {key: [val,...]}."
    result = defaultdict(list)
    for key0, key1 in pairs: #for every key(type=>tuple) in pairs
        result[key0].append(key1)
    return result

class RouteProblem(Problem):
    """A problem to find a route between locations on a `Map`.
    Create a problem with RouteProblem(start, goal, map=Map(...)}).
    States are the vertexes in the Map graph; actions are destination states."""
    
    def actions(self, state): 
        """The places neighboring `state`."""
        return self.map.neighbors[state] 
    
    def result(self, state, action):
        """Go to the `action` place, if the map says that is possible."""
        return action if action in self.map.neighbors[state] else state #go to the neibour city or stay at the current city
    
    def action_cost(self, s, action, s1):
        """The distance (cost) to go from s to s1."""
        return self.map.distances[s, s1]
    
    def h(self, node):
        "Straight-line distance between state and the goal."
        locs = self.map.locations
        return straight_line_distance(locs[node.state], locs[self.goal])
    
    
def straight_line_distance(A, B):
    "Straight-line distance between two points."
    return sum(abs(a - b)**2 for (a, b) in zip(A, B)) ** 0.5

# Some specific RouteProblems
romania_links = {('O', 'Z'):  71, ('O', 'S'): 151, ('A', 'Z'): 75, ('A', 'S'): 140, ('A', 'T'): 118, 
     ('L', 'T'): 111, ('L', 'M'):  70, ('D', 'M'): 75, ('C', 'D'): 120, ('C', 'R'): 146, 
     ('C', 'P'): 138, ('R', 'S'):  80, ('F', 'S'): 99, ('B', 'F'): 211, ('B', 'P'): 101, 
     ('B', 'G'):  90, ('B', 'U'):  85, ('H', 'U'): 98, ('E', 'H'):  86, ('U', 'V'): 142, 
     ('I', 'V'):  92, ('I', 'N'):  87, ('P', 'R'): 97} #dictionary

romania_link_same = [('O', 'Z'), ('O', 'S'), ('A', 'Z'), ('A', 'S'), ('A', 'T'), 
     ('L', 'T'), ('L', 'M'), ('D', 'M'), ('C', 'D'), ('C', 'R'), 
     ('C', 'P'), ('R', 'S'), ('F', 'S'), ('B', 'F'), ('B', 'P'), ('B', 'G'), ('B', 'U'), ('H', 'U'),
     ('E', 'H'), ('U', 'V'), ('I', 'V'), ('I', 'N'), ('P', 'R')] # list, same_weighted_link

romania_locations = {'A': ( 76, 497), 'B': (400, 327), 'C': (246, 285), 'D': (160, 296), 'E': (558, 294), 
     'F': (285, 460), 'G': (368, 257), 'H': (548, 355), 'I': (488, 535), 'L': (162, 379),
     'M': (160, 343), 'N': (407, 561), 'O': (117, 580), 'P': (311, 372), 'R': (227, 412),
     'S': (187, 463), 'T': ( 83, 414), 'U': (471, 363), 'V': (535, 473), 'Z': (92, 539)} #dictionary, values are the coordinate of each sity


romania = Map(romania_links,romania_locations)

# ______________________________________________________________________________
# Python code in FAI_Search.ipynb
# 8 Puzzle game



class EightPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board,
    where one of the squares is a blank, trying to reach a goal configuration.
    A board state is represented as a tuple of length 9, where the element at index i 
    represents the tile number at index i, or 0 if for the empty square, e.g. the goal:
        1 2 3
        4 5 6 ==> (1, 2, 3, 4, 5, 6, 7, 8, 0)
        7 8 _
    """

    def __init__(self, initial, goal=(0, 1, 2, 3, 4, 5, 6, 7, 8)):
        assert inversions(initial) % 2 == inversions(goal) % 2 # Parity check, assert expression => if not expression: raise AssertionError
        self.initial, self.goal = initial, goal
    
    def actions(self, state):
        """The indexes of the squares that the blank can move to."""
        moves = ((1, 3),    (0, 2, 4),    (1, 5),
                 (0, 4, 6), (1, 3, 5, 7), (2, 4, 8),
                 (3, 7),    (4, 6, 8),    (7, 5))
        blank = state.index(0)
        return moves[blank]
    
    def result(self, state, action):
        """Swap the blank with the square numbered `action`."""
        s = list(state) #convert the state from tuple into list s
        blank = state.index(0)
        s[action], s[blank] = s[blank], s[action]
        return tuple(s) #convert the state back into tuple
    
    def h1(self, node):
        """The misplaced tiles heuristic."""
        return hamming_distance(node.state, self.goal)
    
    def h2(self, node):
        """The Manhattan heuristic."""
        X = (0, 1, 2, 0, 1, 2, 0, 1, 2) #x coordinate of goal state 0,1,2,3,4,5,6,7,8
        Y = (0, 0, 0, 1, 1, 1, 2, 2, 2) #y coordinate of goal state 0,1,2,3,4,5,6,7,8
        return sum(abs(X[s] - X[g]) + abs(Y[s] - Y[g])
                   for (s, g) in zip(node.state, self.goal) if s != 0)
    
    def h(self, node): return self.h1(node) 
    
    
def hamming_distance(A, B):
    "Number of positions where vectors A and B are different."
    return sum(a != b for a, b in zip(A, B))
    

def inversions(board):
    "The number of times a piece is a smaller number than a following piece."
    return sum((a > b and a != 0 and b != 0) for (a, b) in combinations(board, 2))
    
    
def board8(board, fmt=(3 * '{} {} {}\n')):
    "A string representing an 8-puzzle board"
    return fmt.format(*board).replace('0', '_') # *board => treat board as 9 arguments
    
# ______________________________________________________________________________
# Python code in FAI_Search.ipynb
# evluation

class CountCalls:
    """Delegate all attribute gets to the object, and count them in ._counts"""
    def __init__(self, obj):
        self._object = obj
        self._counts = Counter() #the self._counts will count number of times each methods has been used.
        
    def __getattr__(self, attr):
        "Delegate to the original object, after incrementing a counter."
        self._counts[attr] += 1
        return getattr(self._object, attr)

        
def report(searchers, problems, verbose=True):
    """Show summary statistics for each searcher (and on each problem unless verbose is false)."""
    show_stat = {};
    for searcher in searchers:
        print(searcher.__name__ + ':')
        total_counts = Counter()
        for p in problems:
            prob   = CountCalls(p)
            soln   = searcher(prob) # run problem p with current searcher, 
            counts = prob._counts; 
            counts.update(path_actions=len(soln), path_cost=soln.path_cost) 
            total_counts += counts
            if verbose: report_counts(counts, str(p)[:40])
        report_counts(total_counts, 'TOTAL\n')

        show_stat[searcher.__name__] = [total_counts['result'],total_counts['is_goal'],total_counts['path_actions']]
    return show_stat
        
def report_counts(counts, name):
    """Print one line of the counts report."""
    print('{:9,d} nodes |{:9,d} goal |{:8.0f} path cost |{:8,d} path actions | {}'.format(
          counts['result'], counts['is_goal'], counts['path_cost'], counts['path_actions'], name)) 

import numpy as np
import matplotlib.pyplot as plt


def show_bar(show_stat):
    columns = ('nodes','goal','actions')
    rows = ['%s' % x for x in show_stat.keys() ]
    
    values = np.arange(0, 100, 25)
    value_increment = 5
   
    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0.5, 1, len(show_stat)))
    n_rows = len(show_stat)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.1

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        index = index+0.1
        plt.bar(index, show_stat[rows[row]], bar_width, bottom=0, color=colors[row],edgecolor = 'white')
    
        #y_offset = y_offset + show_stat[rows[row]]
        y_offset =show_stat[rows[row]]
        cell_text.append(['%1.1f' % (x ) for x in y_offset])
    
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::1]
   # cell_text.reverse()
    
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')
    
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("search comparison")
 #   plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('search criteria')

    plt.show()


