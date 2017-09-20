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

def depthFirstSearchHelper(problem, visitedStates):
  # If we already reached the goal, return an empty list.
  if problem.isGoalState(problem.getStartState()):
      return []
  # Search all of the successors.
  successors = problem.getSuccessors(problem.getStartState())
  for s in successors:
      # If we have visited this successor already, we know that is has no solution
      # and can therefore skip it.
      if s in visitedStates:
          continue
      # Add this successor to the list of visited states, so that we know we don't have to visit it again.
      visitedStates.add(s)
      # Set the problem's start state to the successor's state.
      problem.startState = s[0]
      # Search the successor for a solution.
      result = depthFirstSearchHelper(problem, visitedStates)
      # If a valid solution has been returned, combine it with the successor's action
      # and return it.
      if result is not False:
           return [s[1]] + result
  # If no solution could be found, return False.
  return False

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
  successors = problem.getSuccessors(problem.getStartState())
  print successors
  print problem.getSuccessors(successors[0][0])

  return depthFirstSearchHelper(problem, set())

def breadthFirstSearchHelper(problem, currentCost, frontier, explored, actions):
    """
    Params
    ------
    frontier : [((int, int), int)]
    A list containing a tuple with the destination as the first element and the
    total cost to reach that destination as the second element.

    explored : set((int, int))
    A set containing all destinations that have already been visited.
    """
    # If we reached our goal, no more moves are necessary. Therefore we return an
    # empty list.
    if problem.isGoalState(problem.getStartState()):
        return actions
    # Get all neighbors for the current node and add them to the frontier if they
    # are not already in there.
    successors = problem.getSuccessors(problem.getStartState())
    for s in successors:
        frontier.append( (s[0], actions + [s[1]], currentCost + s[2]) )
    # Sort the frontier lowest cost first.
    frontier = sorted(frontier, key=lambda x: x[2])
    # Take the first element from frontier that is not in explored and search that.
    while True:
        # If there is no valid successor there is no path to goal and we return False.
        if len(frontier) == 0:
            return False
        node = frontier.pop(0)
        if node[0] not in explored:
            break
    # Set the node as the new problem's start state.
    problem.startState = node[0]
    # Add it to the set of explored nodes, so we don't visit it twice.
    explored.add(node[0])
    # And search further down.
    return breadthFirstSearchHelper(problem, node[2], frontier, explored, node[1])

def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  """
  "*** YOUR CODE HERE ***"
  return breadthFirstSearchHelper(problem, 0, [], set(), [])

def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
