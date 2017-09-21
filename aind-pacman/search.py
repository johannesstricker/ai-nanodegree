# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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

def searchHelper(problem, frontier):
    # Initialize frontier with the problem's start state, an empty action list
    # and costs of 0.
    startState = problem.getStartState()
    frontier.push((startState, [], 0))
    # Keep track of already explored nodes.
    explored = set()
    while True:
        if frontier.isEmpty():
            raise Exception, 'No solution found.'
            return False
        # Pop the first node from the frontier.
        node = frontier.pop()
        # If it has been explored already, skip it.
        if node[0] in explored:
            continue
        # If it's the goal state, then return the associated action list.
        if problem.isGoalState(node[0]):
            return node[1]
        # Otherwise add it to the set of explored nodes.
        explored.add(node[0])
        # And add it's children to the frontier.
        successors = problem.getSuccessors(node[0])
        for s in successors:
            # Create an entry with the new state, the action appended to the list
            # and the sum of the costs.
            newState = s[0]
            actions = node[1] + [s[1]]
            costs = node[2] + s[2]
            frontier.push((newState, actions, costs))

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]

  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:

  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  "*** YOUR CODE HERE ***"
  frontier = util.Stack()
  return searchHelper(problem, frontier)

def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  """
  "*** YOUR CODE HERE ***"
  frontier = util.Queue()
  return searchHelper(problem, frontier)

def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  frontier = util.PriorityQueueWithFunction(lambda item: item[2])
  return searchHelper(problem, frontier)

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  frontier = util.PriorityQueueWithFunction(lambda item: heuristic(item[0], problem))
  return searchHelper(problem, frontier)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
