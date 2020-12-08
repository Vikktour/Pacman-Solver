# pacmanAgents.py
# ---------------
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
CS6613I Artificial Intelligence
Victor Zheng
10-07-2020
"""

from pacman import Directions
from game import Agent
from heuristics import *
import random

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        print("state: ", state)
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        print("legal actions: ", legal)
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        print("successors: ", successors)
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in successors]
        print("scored: ", scored)
        # get best choice
        bestScore = min(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

#Node(state,previousNode,theScore,thePrevAction,theDepth)
class Node:
    def __init__(self,state,previousNode,theScore,thePrevAction,theDepth):
        self.state = state
        self.prev = previousNode
        #self.next = next #WE DON'T NEED THIS AT ALL SINCE WE ONLY CARE ABOUT BACKTRACING;note that there can be multiple children, so this is a list
        self.score = theScore
        self.prevAction = thePrevAction
        self.depth = theDepth

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;
    
    #attempt 3 doing BFS using recursion //Got it to work, and it runs away from ghosts, but it gets stuck about mid-way when there's no food near it
    def BFSsearch(self,nodeQueue,leafNodes,winBoolean,winNode):
        if (len(nodeQueue) >= 1):
            #print("node queue length: ", len(nodeQueue)) 
            #winBoolean = False
            #winNode = None
            curNode = nodeQueue[0]
            #print("curnode: ", curNode)
            state = curNode.state
            legal = state.getLegalPacmanActions()
            successors = [(state.generatePacmanSuccessor(action), action) for action in legal]

            
            for s in successors:
                #print("node queue length: ", len(nodeQueue)) 
                #if run out of computation, then save parent to list to compare score later, pop queue and then continue searching
                #if (s == None):
                if (s[0] == None):
                    leafNodes.append(curNode)
                    if (len(nodeQueue) >= 1): #I had to put another check here, idk why we're able to get inside otherwise even with len=0
                        nodeQueue.pop(0)
                    self.BFSsearch(nodeQueue,leafNodes,winBoolean,winNode)
                #basecase1: if winning state then take that path
                elif(s[0].isWin()):
                    winBoolean = True
                    childScore = admissibleHeuristic(s[0])
                    #Node(state,previousNode,theScore,thePrevAction,theDepth)
                    winNode = (s[0],curNode,childScore,s[1],curNode.depth+1)
                    return(leafNodes,winBoolean,winNode)
                #if losing state then don't take the path (i.e. do nothing)
                elif(s[0].isLose()):
                    pass
                #if there are children then recursively search through them
                else:
                    childScore = admissibleHeuristic(s[0])
                    childNode = Node(s[0],curNode,childScore,s[1],curNode.depth+1)
                    nodeQueue.append(childNode)
                    self.BFSsearch(nodeQueue,leafNodes,winBoolean,winNode)

        #basecase2: if nodequeue is empty then return
        return(leafNodes,winBoolean,winNode)   
    
    
    
    def getAction(self, state):
        
        #create node for the root and insert it into queue to be used by BFSsearch function I made
        #Node(state,previousNode,theScore,thePrevAction,theDepth)
        rootNode = Node(state,None,float('inf'),None,0)
        
        nodeQueue = []
        nodeQueue.append(rootNode)
        #BFSsearch(self,nodeQueue,leafNodes,winBoolean,winNode)
        leafNodes,winBoolean,winNode = self.BFSsearch(nodeQueue,[],False,None)
        
        #if there's a winnode then take that path 
        if(winBoolean):
            curDepth = winNode.depth
            curNode = winNode
            while(curDepth > 1):
                curNode = curNode.prev
                curDepth = curDepth-1
            return curNode.prevAction
        #no winning path, so sort the list and get lowest score's move
        else:
            sortedLeafNodes = sorted(leafNodes, key=lambda eachLeafNode:eachLeafNode.score)
            curNode = sortedLeafNodes[0] #best scoring node
            curDepth = curNode.depth
            while(curDepth > 1):
                curNode = curNode.prev
                curDepth = curDepth-1
            return curNode.prevAction
    

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;
    
    
    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP

        #attempt 2 using while loop
        #DFS seems to only find one path (first path it sees it won't look at other ones, so even if I add isLose(), the pacman will still take that path
        #If pacman finds a ghost early it will move a different path
        
        #Keep a list of leaf nodes (or terminal nodes)
        nodeCompareList = []
        
        #For DFS, need to add to stack
        nodeStack = []
        action = None
        depth = 0
        score = depth + admissibleHeuristic(state)
        beginningNode = Node(state,None,score,action,depth) #Node(state,previousNode,theScore,thePrevAction,theDepth)
        nodeStack.append(beginningNode)
        
        winBool = False
        
        while nodeStack:
            currentNode = nodeStack[-1] #last element of list
            state = currentNode.state #get the state of this node
            legal = state.getLegalPacmanActions()
            
            #get next possible states
            successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
            
            #if curNode has no children, append to compareList and pop from stack
            if (successors == None):
                nodeCompareList.append(currentNode)
                nodeStack.pop()
            else:
                sCount = 0
                #for each possible successor state, add to stack and also add to link list
                listOfSuccessorStates = []
                for successor in successors:
                    #if we're out of computations for generatePacmanSuccessor, then don't add to stack
                    if (successor[0] == None):
                        nodeCompareList.append(currentNode)
                        #print("nodeCompareList change 1: ", currentNode.score)
                        nodeStack.pop()
                        break
                    else:
                        listOfSuccessorStates.append(successor[0].isLose())
                        #create the child node
                        sCount = sCount+1
                        #create childNode 
                        childState = successor[0]
                        childDepth = currentNode.depth + 1
                        childScore = childDepth + admissibleHeuristic(childState) #without the childDepth, the pacman gets stuck at the start
                        #childScore = admissibleHeuristic(childState)
                        childPrevAction = successor[1]
                        childNode = Node(childState,currentNode,childScore, childPrevAction,childDepth) #Node(state,previousNode,theScore,thePrevAction,theDepth)
                        
                        
                        #if win state, take that path
                        if (childNode.state.isWin()):
                            #print("Found a win state")
                            winBool = True
                            winNode = childNode
                            break
                        
                        #if lose state don't take it (i.e. do nothing, unless all children are lose states)
                        elif (childNode.state.isLose()):
                            #print("Found a loss state")
                            #print("child score: ", childNode.score)
                            #break #trying this: if lose state then don't use the children anymore #didn't work, the pacman only stopped moving when ghost is near it
                            #if all children have loss states then the currentNode is a leaf node
                            if (sCount == len(successors)):
                                #print("last child is loss state, so appending parent for consideration")
                                #break #trying this: if lose state then don't use the children anymore
                                nodeCompareList.append(currentNode)
                                #print("nodeCompareList change 2: ", currentNode.score)
                        #child is fine
                        else:
                            nodeStack.append(childNode)
    
                    #print("listOfSuccessorStates isLose: ", listOfSuccessorStates)
                
        #if win state is found, then use winNode
        if(winBool):
            curNode = winNode
        #if didn't find a win state then use the compare list to find best score
        else:
            
            #Now compare all the "leaf" nodes in nodeCompareList
            bestScore = float('inf')
            scoreList = [] #for printing and checking what's in it
            for idx,n in enumerate(nodeCompareList):
                scoreList.append(n.score)
                if bestScore > n.score:
                    bestScore = n.score
                    bestScoreIdx = idx
            #print("Score of leafnodes: ", scoreList)        
                
            #take the best leaf node and then iterate back to the 1st level nodes (where the move of the pacman should be made)
            curNode = nodeCompareList[bestScoreIdx]
            
        curDepth = curNode.depth
        #print("curDepth before while loop: ", curDepth)
        #print("action to get to curNode before while loop: ", curNode.prevAction)
        while curDepth > 1:
            #print("curDepth: ", curDepth)
            #print("action to get to curNode: ", curNode.prevAction)
            curDepth = curDepth -1
            curNode = curNode.prev
        
        actionToTake = curNode.prevAction
        #print("actionToTake: ",actionToTake)
        
        return actionToTake
            


class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        g = 0 #initialize the cost to get to current state
        return;
        
    #Attempt 3 using recursion, since while loop made my pacman stuck
    #This ended up running similarly as BFSsearch
    #Note that nodeQueue in this case is a priority queue for nodes
    def ASTARsearch(self,nodeQueue,leafNodes,winBoolean,winNode):
        if (len(nodeQueue) >= 1):
            #print("node queue length: ", len(nodeQueue)) 
            #winBoolean = False
            #winNode = None
            curNode = nodeQueue[0]
            #print("curnode: ", curNode)
            state = curNode.state
            legal = state.getLegalPacmanActions()
            successors = [(state.generatePacmanSuccessor(action), action) for action in legal]

            
            for s in successors:
                #print("node queue length: ", len(nodeQueue)) 
                #if run out of computation, then save parent to list to compare score later, pop queue and then continue searching
                #if (s == None):
                if (s[0] == None):
                    leafNodes.append(curNode)
                    if (len(nodeQueue) >= 1): #I had to put another check here, idk why we're able to get inside otherwise even with len=0
                        nodeQueue.pop(0)
                    self.ASTARsearch(nodeQueue,leafNodes,winBoolean,winNode)
                #basecase1: if winning state then take that path
                elif(s[0].isWin()):
                    winBoolean = True
                    childDepth = curNode.depth+1
                    childScore = childDepth + admissibleHeuristic(s[0])
                    #Node(state,previousNode,theScore,thePrevAction,theDepth)
                    winNode = (s[0],curNode,childScore,s[1],childDepth)
                    return(leafNodes,winBoolean,winNode)
                #if losing state then don't take the path (i.e. do nothing)
                elif(s[0].isLose()):
                    pass
                #if there are children then recursively search through them
                else:
                    childDepth = curNode.depth+1
                    childScore = childDepth + admissibleHeuristic(s[0])
                    childNode = Node(s[0],curNode,childScore,s[1],childDepth)
                    nodeQueue.append(childNode)
                    
            sortedPriorityQueue = sorted(nodeQueue, key=lambda eachLeafNode:eachLeafNode.score)
            self.ASTARsearch(sortedPriorityQueue,leafNodes,winBoolean,winNode) #do another ASTARsearch only after appending all child nodes

        #basecase2: if nodequeue is empty then return
        return(leafNodes,winBoolean,winNode)      
    
    def getAction(self, state):
        
        #create node for the root and insert it into queue to be used by BFSsearch function I made
        #Node(state,previousNode,theScore,thePrevAction,theDepth)
        rootNode = Node(state,None,float('inf'),None,0)
        
        nodeQueue = []
        nodeQueue.append(rootNode)
        #BFSsearch(self,nodeQueue,leafNodes,winBoolean,winNode)
        leafNodes,winBoolean,winNode = self.ASTARsearch(nodeQueue,[],False,None)
        
        #if there's a winnode then take that path 
        if(winBoolean):
            curDepth = winNode.depth
            curNode = winNode
            while(curDepth > 1):
                curNode = curNode.prev
                curDepth = curDepth-1
            return curNode.prevAction
        #no winning path, so sort the list and get lowest score's move
        else:
            sortedLeafNodes = sorted(leafNodes, key=lambda eachLeafNode:eachLeafNode.score)
            curNode = sortedLeafNodes[0] #best scoring node
            curDepth = curNode.depth
            while(curDepth > 1):
                curNode = curNode.prev
                curDepth = curDepth-1
            return curNode.prevAction


