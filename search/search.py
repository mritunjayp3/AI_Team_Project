# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def common_method(problem, search_type, heuristic=nullHeuristic):
    dfs_or_bfs = search_type == 'dfs' or search_type == 'bfs'
    ucs_or_astar = search_type == 'ucs' or search_type == 'astar'
    if search_type == 'dfs':
        data_structure = util.Stack()
    elif search_type == 'bfs':
        data_structure = util.Queue()
    else:
        data_structure = util.PriorityQueue()

    visited = []
    actions_to_return = []
    start_state = problem.getStartState()

    if problem.isGoalState(start_state):
        return []

    if dfs_or_bfs:
        data_structure.push((start_state, actions_to_return))
    elif ucs_or_astar:
        data_structure.push((start_state, actions_to_return, 0), 0)

    while not data_structure.isEmpty():
        if dfs_or_bfs:
            current_state, actions_to_return = data_structure.pop()
        elif ucs_or_astar:
            current_state, actions_to_return, parent_cost = data_structure.pop()

        if current_state not in visited:
            visited.append(current_state)
            if problem.isGoalState(current_state):
                return actions_to_return

            for next_state, action, cost in problem.getSuccessors(current_state):
                updated_action = actions_to_return + [action]
                if search_type == 'ucs':
                    updated_priority = cost + parent_cost
                    data_structure.push((next_state, updated_action, updated_priority), updated_priority)
                elif search_type == 'astar':
                    updated_priority = cost + parent_cost
                    with_heuristic_cost = updated_priority + heuristic(next_state, problem)
                    data_structure.push((next_state, updated_action, updated_priority), with_heuristic_cost)
                else:
                    data_structure.push((next_state, updated_action))


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    return common_method(problem, 'dfs')


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return common_method(problem, 'bfs')


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return common_method(problem, 'ucs')


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return common_method(problem, 'astar', heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
