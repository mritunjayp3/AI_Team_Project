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


def meetInMiddle(problem, heuristic=nullHeuristic):
    """
        This function implements the 'Meet in the Middle' Bi-directional heuristic search algorithm as introduced in the paper,
        "Bidirectional Search That Is Guaranteed to Meet in the Middle" by Robert C. Holte,Ariel Felner, Guni Sharon and Nathan R. Sturtevant;
        In the Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16)

    """    

    print("Meet in the middle Algorithm Initiated")


    """ Importing the required libraries """
    import time
    import util    
    import heapq

    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n=Directions.NORTH
    e=Directions.EAST

    actionDictionary={'North':n, 'South':s, 'East':e, 'West':w}



    def manhattanHeuristic(position, goal):
        "The Manhattan distance heuristic for a PositionSearchProblem"
        xy1 = position
        xy2 = goal
        return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )

    def backtrace(parentChild, initialState, goalNode):
        """ 
        Function to backtrace the actions from initialState to the GoalNode.
        It starts from the goalNode and traces back to the initialNode using the successive parentNode link
        """ 
        actions_=[]                                         #Initializing an empty list to store the actions

        currentState=goalNode                               #Initializing a currentState variable, which shall act as counter till the intialState is reached

        while(currentState!=initialState):
            """ Backtracing is done until we reach the initial state"""
            actions_.insert(0, parentChild[currentState][1])
            currentState=parentChild[currentState][0]       #Updating the currentState to the Parent State
        
        return actions_                                     #Returning the actions_ list


    def min_fValue(allNodes, initialState, parentChildLink, goalState, searchDirection):
        """ This function return the minimum 'f' valuesin a priority queue
            
            - allNode is the list of all nodes in the priority queue along with its priority values
            - initialState has its usual meaning. Note: In forward direction initialState will be true initial state in the search space and in the backward direction, initial state will be the goal state
            - parentChildLink is a dictionary. For more info on it check the description of parentChildForward and parentChildBackward
            - goalState is the final state in the search space depending on the direction
            - searchDirection is the  is the direction of the search

        """

        allFValues=[]
        if(searchDirection=='Forward'):
            for node in allNodes:
                seqAction=backtrace(parentChildLink, initialState, node[2])
                Cost_= problem.getCostOfActions(seqAction) + manhattanHeuristic (node[2], goalState)
                allFValues.append(Cost_)
        else:
            for node in allNodes:
                seqAction=backtrace(parentChildLink, initialState, node[2])
                Cost_= problem.getCostOfActionsBackward(seqAction, initialState) + manhattanHeuristic (node[2], goalState)
                allFValues.append(Cost_) 

        return(min(allFValues))


    def min_gValue(allNodes, initialState, parentChildLink, searchDirection):

        """ This function return the minimum 'g' valuesin a priority queue
            
            - allNode is the list of all nodes in the priority queue along with its priority values
            - initialState has its usual meaning. Note: In forward direction initialState will be true initial state in the search space and in the backward direction, initial state will be the goal state
            - parentChildLink is a dictionary. For more info on it check the description of parentChildForward and parentChildBackward
            - searchDirection is the  is the direction of the search

        """
        allGValues=[]

        if(searchDirection=='Forward'):
            for node in allNodes:
                seqAction=backtrace(parentChildLink, initialState, node[2])
                Cost_= problem.getCostOfActions(seqAction)
                allGValues.append(Cost_)

        else:
            for node in allNodes:
                seqAction=backtrace(parentChildLink, initialState, node[2])
                Cost_= problem.getCostOfActionsBackward(seqAction, initialState)
                allGValues.append(Cost_)
        return(min(allGValues))


    
    frontierStatesForward=util.PriorityQueue()              # A priority list data structure to keep the frontier nodes for exploring from start state to the goal state.
                                                            #Its defined in the util. Nodes with least cost are prioritized first.

    frontierStatesBackward=util.PriorityQueue()             #A priority list data structure to keep the frontier nodes for exploring from goal state to the start state.
                                                            #Its defined in the util. Nodes with least cost are prioritized first.

    exploredStatesForward = []                              #List of all the states that have been explored/expanded while exploring from start state to goal state direction
    exploredStatesBackward = []                             #List of all the states that have been explored/expanded while exploring from goal state to start state direction

    initialState_Forward = problem.getStartState()          #Getting the intial State of the search problem for the forward direction search
    initialState_Backward = problem.goal                    #Getting the initial State of the search problem for the backward direction search

    frontierStatesForward.push(initialState_Forward,manhattanHeuristic(initialState_Forward, initialState_Backward))      #Appending the intial state into the frontierStatesForward list
    frontierStatesBackward.push(initialState_Backward,manhattanHeuristic(initialState_Backward, initialState_Forward))    #Appending the intial state into the frontierStatesBackward list
    

    
    U = float('inf')    # Initialzing the cost to the the goal state
    epsilon = 1         # Cost of the minimum edge in state space


    parentChildForward={}
    parentChildBackward={}
    """
        Description for the parentChild dictionary
        A dictionary to link the child node with the parent node, which will be used to backtrace the actions.
        - Keys in the dictionary represent the state represented by the child node.
        - The values are a set that represent the 'parent node state' and 'action' required to get from parent state to the respective child state.
    """

    while(1):

        # Checking if both the Priority Queues are empty.
        # If both are empty then no such path exist.
        if(frontierStatesForward.isEmpty() and frontierStatesBackward.isEmpty()):
            print('Both frontierStatesForward and frontierStatesBackward are empty.')
            print('No path from start to goal state exist.')
            return None

        else:

            # Getting the minimum 'f' values in each direction
            fmin_Forward = min_fValue(frontierStatesForward.heap, initialState_Forward, parentChildForward, initialState_Backward, 'Forward')
            fmin_Backward = min_fValue(frontierStatesBackward.heap, initialState_Backward, parentChildBackward, initialState_Forward, 'Backward')


            # Getting the minimum 'g' values in each direction
            gmin_Forward = min_gValue(frontierStatesForward.heap, initialState_Forward, parentChildForward, 'Forward')
            gmin_Backward = min_gValue(frontierStatesBackward.heap, initialState_Backward, parentChildBackward, 'Backward')


            # Fetching the highest priority values in each search directions
            minPriorityValueinForwardQueue = heapq.nsmallest(1,frontierStatesForward.heap)[0][0]
            minPriorityValueinBackwardQueue = heapq.nsmallest(1, frontierStatesBackward.heap)[0][0]

            
            minValue= min(minPriorityValueinForwardQueue, minPriorityValueinBackwardQueue)

            if(U <= max(minValue, fmin_Forward, fmin_Backward, gmin_Forward + gmin_Backward + epsilon)):

                seqAction1=backtrace(parentChildForward, initialState_Forward, midNode)
                seqAction2=backtrace(parentChildBackward, initialState_Backward, midNode)
                
                print(len(seqAction1))
                print(len(seqAction2))

                seqAction2New=[]

                for action in seqAction2:
                    if(action=='East'):
                        seqAction2New.insert(0,'West')

                    if(action=='West'):
                        seqAction2New.insert(0, 'East')    
                    
                    if(action=='South'):
                        seqAction2New.insert(0, 'North')    
                    
                    if(action=='North'):
                        seqAction2New.insert(0, 'South')    
    
                seqAction=seqAction1 + seqAction2New

                """Return the Action Seqeuence"""
                print("The cost of the path is: " + str(U))
                print("The mid-node is: " + str(midNode))
                return seqAction

                break

            else:

                if(minValue==minPriorityValueinForwardQueue):

                    parentNodeForward = frontierStatesForward.pop()          
                    if(parentNodeForward not in exploredStatesForward):
                        newFrontierNodes=problem.getSuccessors(parentNodeForward)
                        exploredStatesForward.append(parentNodeForward)

                        for childNodes in newFrontierNodes:
                            """Verifying if each child node exists in explored states set or in the frontier set"""
                            if(childNodes[0] not in exploredStatesForward):

                                """We check if the childNode is already present in the childParent Dictionary"""
                                if(childNodes[0] not in parentChildForward.keys()):

                                    """Linking the ChildNode with the ParentNodeForward and keeping it in a dictionary"""

                                    parentChildForward[childNodes[0]]=(parentNodeForward, childNodes[1])

                                    """Calling the backtracing function to get the the sequence of actions which are required to go
                                       from initial state to the current childNode state
                                    """
                                    seqAction=backtrace(parentChildForward, initialState_Forward, childNodes[0])

                                    Cost_=problem.getCostOfActions(seqAction) + manhattanHeuristic(childNodes[0], initialState_Backward)

                                    priorityValue= max(Cost_, 2*problem.getCostOfActions(seqAction))

                                    """Adding the childNode state to the frontier list along with the Cost_ to reach the ChildNode"""
                                    frontierStatesForward.push(childNodes[0], priorityValue)


                                else:
                                    """ If the childNode is already present, we update the child:Parent link only if the current cost is lesser than the past cost"""

                                    """Getting the past cost"""
                                    Past_Cost=problem.getCostOfActions(backtrace(parentChildForward, initialState_Forward, childNodes[0]))
                                    """Getting the current cost"""
                                    Cost_=problem.getCostOfActions(backtrace(parentChildForward, initialState_Forward, parentNodeForward))+childNodes[2]

                                    """Adding the heuristic cost"""
                                    Cost_= Cost_ + manhattanHeuristic(childNodes[0], initialState_Backward)

                                    if(Past_Cost > Cost_):
                                        """Adding the childNode to the frontier list"""
                                        frontierStatesForward.push(childNodes[0],Cost_)
                                        """Updating the Child:Parent link in the parentChildForward linkup"""
                                        parentChildForward[childNodes[0]]=(parentNodeForward, childNodes[1])

                                        priorityValue= max(Cost_, 2*problem.getCostOfActions(seqAction))    
                                        frontierStatesForward.push(childNodes[0], priorityValue)

                        
                                if(childNodes[0] in exploredStatesBackward):
                                    costofStartStatetoNode = problem.getCostOfActions(backtrace(parentChildForward, initialState_Forward, childNodes[0]))
                                    costofGoalStatetoNode  = problem.getCostOfActionsBackward(backtrace(parentChildBackward, initialState_Backward, childNodes[0]), initialState_Backward)
                                    
                                    # Update U 
                                    U = min(U, costofStartStatetoNode+costofGoalStatetoNode)
                                    if((costofStartStatetoNode+costofGoalStatetoNode)==U):
                                        midNode = childNodes[0]

                else:
                

                    parentNodeBackward = frontierStatesBackward.pop()          
                    if(parentNodeBackward not in exploredStatesBackward):
                        newFrontierNodes=problem.getSuccessors(parentNodeBackward)
                        exploredStatesBackward.append(parentNodeBackward)

                        for childNodes in newFrontierNodes:
                            """Verifying if each child node exists in explored states set or in the frontier set"""
                            if(childNodes[0] not in exploredStatesBackward):

                                """We check if the childNode is already present in the childParent Dictionary"""
                                if(childNodes[0] not in parentChildBackward.keys()):

                                    """Linking the ChildNode with the parentNodeBackward and keeping it in a dictionary"""
                                    parentChildBackward[childNodes[0]]=(parentNodeBackward, childNodes[1])

                                    """Calling the backtracing function to get the the sequence of actions which are required to go
                                       from initial state to the current childNode state
                                    """
                                    seqAction=backtrace(parentChildBackward, initialState_Backward, childNodes[0])

                                    Cost_=problem.getCostOfActionsBackward(seqAction,initialState_Backward) + manhattanHeuristic(childNodes[0], initialState_Forward)

                                    priorityValue= max(Cost_, 2*problem.getCostOfActionsBackward(seqAction, initialState_Backward))

                                   
                                    """Adding the childNode state to the frontier list along with the Cost_ to reach the ChildNode"""
                                    frontierStatesBackward.push(childNodes[0], priorityValue)


                                else:
                                    """ If the childNode is already present, we update the child:Parent link only if the current cost is lesser than the past cost"""

                                    """Getting the past cost"""
                                    Past_Cost=problem.getCostOfActionsBackward(backtrace(parentChildBackward, initialState_Backward, childNodes[0]), initialState_Backward)
                                    """Getting the current cost"""
                                    Cost_=problem.getCostOfActionsBackward(backtrace(parentChildBackward, initialState_Backward, parentNodeBackward), initialState_Backward)+childNodes[2]

                                    """Adding the heuristic cost"""
                                    Cost_= Cost_ + manhattanHeuristic(childNodes[0], initialState_Forward)

                                    if(Past_Cost > Cost_):
                                        """Adding the childNode to the frontier list"""
                                        frontierStatesBackward.push(childNodes[0],Cost_)
                                        """Updating the Child:Parent link in the parentChildBackward linkup"""
                                        parentChildBackward[childNodes[0]]=(parentNodeBackward, childNodes[1])

                                        priorityValue= max(Cost_, 2*problem.getCostOfActionsBackward(seqAction), initialState_Backward)
  
                                        frontierStatesBackward.push(childNodes[0], priorityValue)

                            
                                if(childNodes[0] in exploredStatesForward):
                                    
                                    costofStartStatetoNode = problem.getCostOfActions(backtrace(parentChildForward, initialState_Forward, childNodes[0]))
                                    costofGoalStatetoNode  = problem.getCostOfActionsBackward(backtrace(parentChildBackward, initialState_Backward, childNodes[0]), initialState_Backward)
                                    
                                    # Update U 
                                    U = min(U, costofStartStatetoNode+costofGoalStatetoNode)
                                    if((costofStartStatetoNode+costofGoalStatetoNode)==U):
                                        midNode = childNodes[0]
                

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
mm = meetInMiddle