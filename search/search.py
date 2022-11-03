# THIS  CODE  WAS MY OWN WORK , IT WAS  WRITTEN  WITHOUT  CONSULTING  ANY
# SOURCES  OUTSIDE  OF THOSE  APPROVED  BY THE  INSTRUCTOR. Daniel Ku.
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def graphSearch(problem, dataStructure):
    # Push the root node into the data structure
    dataStructure.push([(problem.getStartState(), "Stop", 0)])

    # Initialize list of visited nodes on graph to a list
    visited = []

    while not dataStructure.isEmpty():
        path = dataStructure.pop()

        currentState = path[-1][0]

        # Check if currentState is the goal state & return the path to get to the goal state
        if problem.isGoalState(currentState):
            return [x[1] for x in path][1:]
        # check if the currentState has been visited
        if currentState not in visited:
            # if not visited, visit and add to visited list
            visited.append(currentState)
        
            # traverse through all successors for the currentState
            for successor in problem.getSuccessors(currentState):
                # check if successor is visited
                if successor[0] not in visited:
                    # copy whole path
                    successorPath = path[:]
                    # set path of successor to the parent's path + successor node
                    successorPath.append(successor)
                    dataStructure.push(successorPath)
    # if everything fails, i.e. no path found, return empty list
    return []



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    # initialize empty stack for DFS
    stack = util.Stack()
    return graphSearch(problem, stack)


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    # initialize empty queue for BFS
    queue = util.Queue()
    return graphSearch(problem, queue)

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    # get the action in the tuple in the path, then get the cost of the action
    cost = lambda path: problem.getCostOfActions([x[1] for x in path][1:])

    # initialize Priority Queue that sorts according to cost
    priorityQueue = util.PriorityQueueWithFunction(cost)
    return graphSearch(problem, priorityQueue)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    # a* search chooses nodes from fringe by smallest sum
    # f(x) = g(x) + h(x) where g(x) is cost defined in UCS and h(x) is heuristic
    cost = lambda path: problem.getCostOfActions([x[1] for x in path][1:]) + heuristic(path[-1][0], problem)

    # initialize Priority Queue that sorts according to f(x)
    pq = util.PriorityQueueWithFunction(cost)
    return graphSearch(problem, pq)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
