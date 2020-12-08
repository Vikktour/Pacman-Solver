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
11-03-2020
"""

from pacman import Directions
from game import Agent
from heuristics import *
import random
import math

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

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];


class HillClimberAgent(Agent):
    #create multiple sequences with the 50% mutation and keep comparing until you run out of successor calls
    #compare score(5th move, currentState)
    #after getAction() you take the 1st move and then reset the sequence to a new random one and run the same process
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.seqLength = 5
        
        return;
    
    #list of Directions.STOP
    def generateStopSequence(self,state):
        actionList = [];
        #create an action sequence of length 5, start out with random 5 moves
        for i in range(0,self.seqLength):
            actionList.append(Directions.STOP);
        return actionList
    
    def generateSequence(self,state):
        actionList = self.generateStopSequence(state)
        #create an action sequence of length 5, start out with random 5 moves
        possible = state.getAllPossibleActions();
        for i in range(0,len(actionList)):
            actionList[i] = possible[random.randint(0,len(possible)-1)];    
        return actionList
        
    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        
        #generate a sequence (starting point of hill climb)
        currentSequence = self.generateSequence(state)
        startState = state
        
        currentSequenceLose = True
        #Keep generating currentSequence until the whole sequence doesn't run into lose state
        while (currentSequenceLose == True):
            
            currentStateC = state
            for i in range(0,len(currentSequence)):
                currentStateC = currentStateC.generatePacmanSuccessor(currentSequence[i])
                #print("currentSequence: ", currentSequence)
                #print("currentStateC: ", currentStateC)
                #if out of successor calls while still finding a non-losing sequence to start hillclimb then we lose
                if(currentStateC == None):
                    print("All currentSequences generated leads to lose")
                    return Directions.STOP
                
                elif(currentStateC.isLose()):
                        #if currentActionList is a lose sequence, then make a new one and restart checking
                        #self.seqLength = 1 #I tried to change the length of sequence to 1 to make this easier for pacman to escape ghost,
                        #but it seems that the state where it runs to wall and ghost hits it is safe, even though it gets eaten
                        print("-------------------------------------------------------------")
                        print("Lose state exists, so make new currentSequence")
                        print("Current Sequence: ", currentSequence)
                        currentSequence = self.generateSequence(state)
                        print("Current Sequence new: ", currentSequence)
                        print("-------------------------------------------------------------")
                        break
                elif(currentStateC.isWin()):
                    return currentSequence[0]
                    
                if(i == len(currentSequence)-1):
                    #on last iteration and none of the states are: isWin isLose or None, then we have our safe starting sequence
                    currentSequenceLose = False
                    #print("This current Sequence has no lose states")
                    #print("currentSequence: ", currentSequence)
        print("successfully generated a good currentSequence (no loss states detected)")
        print("currentSequence being used: ", currentSequence)
        #self.seqLength = 5
        """
        #making sure that currentSequence does not have lose state, but apparently there is...
        #I feel like the second time check doesn't work bc generateSuccessor makes a new state (despite calling the same action) than in the testing before
        currentStateC = state
        for i in range(0,len(currentSequence)):
            currentStateC = currentStateC = currentStateC.generatePacmanSuccessor(currentSequence[i])
            if(currentStateC == None):
                print("All currentSequences generated leads to lose")
            elif(currentStateC.isLose()):
                print("this currentSequenccec has lose state")
            elif(currentStateC.isWin()):
                print("win state")
                return currentSequence[0]
        """
        
        #keep creating mutatedSequences until out of successorcalls
        breakWhile = False
        while(breakWhile == False):
            #currentSequenceIsGuaranteedNotLose = False #to be more efficient, we don't have to generateSuccessor for currentSequence to do checks, but for simplilcity, I'll not be efficient
            currentSequenceLose = False
            mutatedSequenceLose = False
            currentSequenceWin = False
            mutatedSequenceWin = False
            currentStateC = state
            currentStateM = state
            
            #Create another sequence that is mutated off of the currentSequence
            #mutatedSequence = self.generateStopSequence(state) #this makes pacman a bit more food hungry (good)
            mutatedSequence = currentSequence
            for i in range(0,len(currentSequence)):
                randN = random.uniform(0, 1) #generate random number [0,1)
                #50% that an action will be changed to a random action
                if (randN < 0.5):
                    possible = state.getAllPossibleActions(); #Available actions are: North South East West
                    randomaction = possible[random.randint(0,len(possible)-1)]
                    mutatedSequence[i] = randomaction
            
            #compare current sequence with mutated sequence to see which is better
            for i in range(0,len(currentSequence)):
                #currentSequence
                currentStateC = currentStateC.generatePacmanSuccessor(currentSequence[i])
                #mutatedSequence
                currentStateM = currentStateM.generatePacmanSuccessor(mutatedSequence[i])
                
                #check for win/lose/None
                if(currentStateC == None):
                    if(currentSequenceLose == False):
                        #print("return 2")
                        return currentSequence[0]
                    else:
                        print("Error: All sequences generated, despite using up successor calls, ends up losing")
                    breakWhile = True
                    break
                elif(currentStateC.isLose()):
                    currentSequenceLose = True #need to ensure this never happens by creating a currentSequence that never loses before going into this while loop
                    print("currentStateC.isLose()")
                    #return Directions.STOP
                    return currentSequence[0]
                elif(currentStateC.isWin()):
                    currentSequenceWin = True
                    print("return 1")
                    return currentSequence[0]
                
                
                if(currentStateM == None):
                    #if out of successor calls then currentSequence is returned because it's based on previous iterations of comparisons
                    if(currentSequenceLose == False): #checking in case this is the first iteration
                        print("return 4")
                        return currentSequence[0]
                    else:
                        print("Error: All sequences generated, despite using up successor calls, ends up losing")
                    breakWhile = True
                    break
                elif(currentStateM.isLose()):
                    mutatedSequenceLose = True
                    break #move on to next mutatedSequence
                elif(currentStateM.isWin()):
                    mutatedSequenceWin = True
                    print("return 3")
                    return mutatedSequence[0]
                
                    
                #if at the last action of the sequence, then compare the score and take the higher one
                if(i == len(currentSequence)-1):
                    scoreC = gameEvaluation(startState, currentStateC)
                    scoreM = gameEvaluation(startState, currentStateM)
                    
                    #change currentSequence to be mutatedSequence if the latter is better
                    if(scoreM > scoreC):
                        currentSequence = mutatedSequence
                        
                    #currentSequenceIsGuaranteedNotLose = True
                
        print("default return at end of function")
        return currentSequence[0] #testing

#create a class for members of the population from GeneticAgent
class populationMember:
    def __init__(self):
        self.sequence = None
        self.score = None #A (-1) score means the member will result lose
        
        self.rank = None
        self.selectionProb = None
        self.offsetProb = None #probability with offset for rolling purposes (e.g. given probs: (0.55,0.2,0.15,0.1) we have dice rolled to x<0.55 for sequence 1, 0.55<x<0.75 for sequence 2, etc)

#My implementation turned out to be not that great; pacman hesitates when the ghost is nearby. It also doesn't try to aim for food. I noticed that the scores when printed are pretty low (e.g. 0.00,0.01,0.02,0.03)
class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.seqLength = 5 #5 chromosomes per creature
        self.populationSize = 8 #8 creatures in our population
        self.currentBestSeq = []
        return;
    
    def createPopulation(self,state):
        self.population = []
        for i in range(0,self.populationSize):
            sequence = self.generateSequence(state)
            member = populationMember()
            member.sequence = sequence
            self.population.append(member)
        
        self.currentBestSeq = self.population[0].sequence
        
        return;
    
    #list of Directions.STOP
    def generateStopSequence(self,state):
        actionList = [];
        #create an action sequence of length 5, start out with random 5 moves
        for i in range(0,self.seqLength):
            actionList.append(Directions.STOP);
        return actionList
    
    #generate a sequence
    def generateSequence(self,state):
        actionList = self.generateStopSequence(state)
        #create an action sequence of length 5, start out with random 5 moves
        possible = state.getAllPossibleActions();
        for i in range(0,len(actionList)):
            actionList[i] = possible[random.randint(0,len(possible)-1)];    
        return actionList

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        
        self.createPopulation(state)
        #print("population: ",self.population)
        #print("population[0].sequence: ",self.population[0].sequence)
        
        startState = state
        #FITNESS RANKING
        
        while(True):
            #FITNESS ASSESSMENT
            #for each member of the population, update the scores
            for i in range(0,self.populationSize):
                currentSequence = self.population[i].sequence
                currentState = state
                for j in range(0,self.seqLength):
                    currentState = currentState.generatePacmanSuccessor(currentSequence[j])
                    
                    #if out of successor calls then return the best sequence's move
                    if(currentState == None):
                        return self.currentBestSeq[0]
                    #if this sequence has a loss state then we can't evaluate it, put it at the end
                    elif(currentState.isLose()):
                        self.population[i].score = -1
                        break
                    elif(currentState.isWin()):
                        #population[i].score = 10000 #some big number #don't need this because we just pick that sequence
                        return currentSequence[0]
                    #at the end of the sequence, we compute the score
                    elif(j == self.seqLength-1):
                        currentScore = gameEvaluation(startState, currentState)
                        self.population[i].score = currentScore
                        #print("updating score: ", currentScore)
                    
        
            #sort the members according to score
            sortedPopulation = sorted(self.population, key=lambda eachMember:eachMember.score, reverse=True) #sort in descending order of score
            self.population = sortedPopulation
            self.currentBestSeq = sortedPopulation[0].sequence
            #verify that the population is sorted (TRUE)
            #print("population[0].score: ", self.population[0].score)
            #print("population[1].score: ", self.population[1].score)
            #print("population[2].score: ", self.population[2].score)
            
            sumOfRanks = 0 #this is for fitness function later
            #rank each member according to score (fitness ranking)
            for i in range(0,self.populationSize):
                self.population[i].rank = self.populationSize - i #8,7,6,...,2,1
                sumOfRanks = sumOfRanks + self.population[i].rank
            #verify the ranks (TRUE)
            #for member in self.population:
            #    print(member.rank), #the comma makes it print on same line
            
            #calculate fitness proportionate selection of each member according to rank
            offset = 0#probability offset to be used for the number roll
            for i in range(0,self.populationSize):
                self.population[i].selectionProb = float(self.population[i].rank) / float(sumOfRanks)
                self.population[i].offsetProb = self.population[i].selectionProb + offset
                offset = offset + self.population[i].selectionProb
            #verify the selectionProb
            #totalProb = 0
            #for member in self.population:
            #    totalProb = totalProb + member.selectionProb
            #    print("selecionProb of all members: ",member.selectionProb) #adding a comma at the end makes it print on same line 
            #    print("offsetProb of all members: ",member.offsetProb)
            #print("totalProb should be 1. We got: ", totalProb)
            
            #print scores for debug
            #for member in self.population:
            #    print("Member score: ",member.score),
            
            #SELECT PARENTS
            #Select 4 pairs of parents (parents can be repeated) based on Rank Selection (selection probability)
            numPairs = self.populationSize/2 #note that populationSize should be even otherwise numPairs will round down
            
            #For each pair we will make (4 pairs for this hw), roll random num and then match it to parent
            pairArray = []
            for pair in range(0,numPairs):
                randNx = random.uniform(0,1) #generate random number [0,1), for parentX
                randNy = random.uniform(0,1) #generate random number [0,1), for parentY
                parentXSet = False
                parentYSet = False
                for parent in range(0,self.populationSize):
                    if(parentXSet == False):
                        if(randNx < self.population[parent].offsetProb):
                            parentX = self.population[parent]
                            parentXSet = True
                    if(parentYSet == False):
                        if(randNy < self.population[parent].offsetProb):
                            parentY = self.population[parent]
                    if(parentXSet and parentYSet):
                        break
                        
                parentPair = [parentX,parentY]
                pairArray.append(parentPair)
            
            #validate that there are 4 pairs of parents
            #print("pairArray: ", pairArray)
            
            
            #CROSSOVER (Breeding)
            
            newPopulation = []
            for eachPair in pairArray:
                parentX = eachPair[0]
                parentY = eachPair[1]
                
                child1 = populationMember()
                child2 = populationMember()
                #random test: <=70% do crossover, >70% keep parents
                randN = random.uniform(0,1)
                #Crossover
                if(randN <= 0.7):
                    childSeq1 = []
                    childSeq2 = []
                    #<50% donate X, otherwise donate Y
                    for i in range(0,self.seqLength):
                        
                        #child 1
                        rand1 = random.uniform(0,1)
                        if(rand1<0.5):
                            childSeq1.append(parentX.sequence[i])
                        else: #rand1>0.5
                            childSeq1.append(parentY.sequence[i])
                        
                        #child 2
                        rand2 = random.uniform(0,1)
                        if(rand2<0.5):
                            childSeq2.append(parentX.sequence[i])
                        else: #rand2>0.5
                            childSeq2.append(parentY.sequence[i])
                    
                    child1.sequence = childSeq1
                    child2.sequence = childSeq2
                    newPopulation.append(child1)
                    newPopulation.append(child2)
                
                #Keep parents
                elif(randN > 0.7):
                    newPopulation.append(parentX)
                    newPopulation.append(parentY)
            
            #verify that the new population is made
            #print("--- NEW POPULATION ---")
            #for eachMember in newPopulation:
            #    print("Member sequence: ",eachMember.sequence)
            
            #MUTATION
            #for every member, if <=10%, then randomly choose an action in the sequence to mutate to a random action
            i = 0
            for member in newPopulation:
                randN = random.uniform(0,1)
                if(randN <= 0.1):
                    mutationIdx = random.randint(0,self.seqLength-1) #index of sequence to mutate
                    possible = state.getAllPossibleActions()
                    mutationMove = possible[random.randint(0,len(possible)-1)]; #move to mutate into
                    #member.sequence[mutationIdx] = Directions.STOP #this doesn't update the newPopulation array, since member is a copy
                    newPopulation[i].sequence[mutationIdx] = mutationMove
                    i = i + 1
            #Verify that the mutation occurred (insert Directions.STOP)
            #for member in newPopulation:
            #    print("Verify MemberSequence of new population: ", member.sequence)
            
            #update population
            self.population = newPopulation
            
        
        return self.currentBestSeq[0]

#Node class defined for MCTS
class Node:
    #some of these variables were not needed in the final version of my MCTS
    def __init__(self=None,xBar=None,nj=None,prevAction=None,depth=None,prev=None,index=None,hasChildren=False,children=None,isFullyExpanded=False,state=None,isTerminal=False,Q=0): #all variables are set to None by default
        self.xBar = xBar #Exploitation value (average value of rewards thus far that were discovered from the children of this node)
        self.nj = nj #number of times this node was visited
        self.prevAction = prevAction #the previous action that got to this node
        self.depth = depth #depth level of this Node, 0 being the top node
        self.prev = prev #previous Node
        self.index = index #for keeping track of where the node is in an array
        self.hasChildren = hasChildren #for moving on in selection
        self.children = children #a list of children of this Node
        self.isFullyExpanded = isFullyExpanded #bool to know if a node is fully expanded yet (i.e. if #children = #legalaction)
        self.state = state #Some parts require to save states (e.g. MCTSSearch, Expand). Despite the assignment saying not to save states.
        self.isTerminal = isTerminal #This node is terminal if we hit the isWin or isLose state
        self.Q = Q #cumulative score of this node 

#In general, it managed to avoid ghosts (especially with more iterations), but sometimes it runs to the ghost at the start
class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        #self.NodesVisited = [] #List of nodes visited
        return;
    
    def MCTSSearch(self,s0):
        #print("Starting MCTSSearch")
        #initialize variables
        self.NodesVisited = [] #List of nodes visited
        self.n = 0 #tracking total number of node visits (including revisits)
        
        #root node will be appended at index 0
        v0 = Node(state=s0,index=0) 
        self.NodesVisited.append(v0)
        while(self.ComputationIsNone == False): #if generatePacmanSuccessor is None then break out
            v0 = self.NodesVisited[v0.index] #make sure it's the real v0 from the array
            
            vL = self.TreePolicy(v0)
            #print("------- End of TreePolicy ------")
            counter = 0
            for node in self.NodesVisited:
                #if(node.children == None):
                #    length = 0
                #else:
                #    length = len(node.children)
                #print("Index {0} has {1} children".format(counter,length))
                counter = counter+1
                
            if(self.ComputationIsNone == True):
                break
            delta = self.DefaultPolicy(vL.state)
            #print("------- End of DefaultPolicy ------")
            counter = 0
            for node in self.NodesVisited:
                #if(node.children == None):
                #    length = 0
                #else:
                #    length = len(node.children)
                #print("Index {0} has {1} children".format(counter,length))
                counter = counter+1
            
            if(self.ComputationIsNone == True):
                break
            self.Backup(vL,delta)
            
            #print("------- End of BackUP ------")
            counter = 0
            for node in self.NodesVisited:
                #if(node.children == None):
                #    length = 0
                #else:
                #    length = len(node.children)
                #print("Index {0} has {1} children".format(counter,length))
                counter = counter+1
            
        #Once we run out of computation resources from above, return action with highest amount of visits from the original state
        rootNode = self.NodesVisited[0]
        #find the children with the largest number of visits
        highestVisits = 0
        bestChild = None
        for i in range(0,len(rootNode.children)):
            childVisits = rootNode.children[i].nj
            if (highestVisits < childVisits):
                highestVisits = childVisits
                bestChild = rootNode.children[i]
            
            #error catch if there is no tree
            if(bestChild == None):
                print("There is no child from root node")
                return Directions.STOP
            
        return bestChild.prevAction #will return best action 

    #inputs a node, will expand if not expanded, and if expanded then keep moving down the path until reach a non-expanded node
    def TreePolicy(self,v):
        #print("Starting TreePolicy")
        #print("v.index at TreePolicy: ", v.index)
        while(v.isTerminal == False):
            if (v.isFullyExpanded == False):
                return self.Expand(v)
            else:
                c = 1 #for this assignment we want c to be 1
                v = self.Select(v,c)
            
        return v
    
    #Expand a node with a random legal action (that is not yet taken). Random to avoid bias.
    #Note that I'm also inputting currentState to avoid saving the states in each node // changed my mind and tried to store state to node v
    def Expand(self,v):
        #print("Starting Expand of node v with index {}".format(v.index))
        #print("v.children: ", v.children)
        currentState = v.state
        #print("currentState: ", currentState)
        legalActions = currentState.getLegalPacmanActions()
        takenActions = [] #will store all actions for the current children we have thus far
        #if(len(v.children) != 0): #I used this before when Node( children = []) initializer doesn't work, it will cause the childNode's child to be itself
        if(v.children != None):
            for child in v.children:
                takenActions.append(child.prevAction)
        
        #print("legalActions: ", legalActions)
        #print("takenActions: ", takenActions)        
        #possibleActions = legalActions - takenActions, meaning this gives back actions not yet taken and are legal
        possibleActions = [x for x in legalActions if x not in takenActions] #shady
        #print("possibleActions: ", possibleActions)
        #if all actions are taken then we're done exploring this node
        #if (possibleActions == None):
        #    self.NodesVisited[v.index].isFullyExpanded = True
           
        
        randomMoveIdx = random.randint(0,len(possibleActions)-1)
        actionToTake = possibleActions[randomMoveIdx]
        newChildState = currentState.generatePacmanSuccessor(actionToTake)
        if(currentState == None):
            self.ComputationIsNone = True
            return
        newChild = Node(state=newChildState,nj=1,prevAction=actionToTake,prev=v,hasChildren=False,index=len(self.NodesVisited))
        
        #if(newChild.children == None):
        #    length = 0
        #else:
        #    length = len(newChild.children)
        #print("newChild num of children: ", length)
        
        if (v.children == None):
            v.children = []
        v.children.append(newChild)
        
        self.NodesVisited.append(newChild)
        #if(newChild.children == None):
        #    length = 0
        #else:
        #    length = len(newChild.children)
        #print("newChild num of children2: ", length)
        
        #after this we're done expanding
        #print("Need this to be 1: len(possibleActions): ", len(possibleActions))
        if( len(possibleActions) == 1 ):
            v.isFullyExpanded = True
            updateIndex = v.index
            self.NodesVisited[updateIndex] = v
            """print("NodesVisited[{}] is now fully expanded".format(v.index))"""
            
        #print("------- End of Expand ------")
        #counter = 0
        #for node in self.NodesVisited:
        #    if(node.children == None):
        #        length = 0
        #    else:
        #        length = len(node.children)
        #    print("Index {0} has {1} children".format(counter,length))
        #    counter = counter+1    
            
        
        return newChild
    
    #given a node, select a child path to take
    def Select(self,v,c):
        #print("Starting Select")
        #This assignment specifies to let c=1, so we will input 1 when we call this function
        totalN = self.n#totalN is the total amount of times we visited a node in this current getAction
        curMaxScore = 0
        curMaxIdx = 0
        for i in range(0,len(v.children)):
            curChild = v.children[i]
            #childScore = curChild.xBar + c*math.sqrt(2*math.log(totalN)/curChild.nj)
            childScore = (curChild.Q / curChild.nj) + c*math.sqrt(2*math.log(totalN)/curChild.nj)
            if(curMaxScore < childScore):
                curMaxScore = childScore
                curMaxIdx = i
        
        selectedChild = v.children[curMaxIdx]
        
        #if lose state then isTerminal
        if(selectedChild.state.isLose()):
            selectedChild.isTerminal = True
            self.NodesVisited[selectedChild.index] = selectedChild #update the NodesVisited Array with terminal status
        
        return selectedChild #child with the highest selection score
    
    def DefaultPolicy(self,s):
        #rollout function
        #for this hw, we rollout 5 moves
        #print("Starting DefaultPolicy")
        currentState = s
        #if none state then we're out of computation resources, break
        if (currentState == None):
            self.ComputationIsNone = True
            return None
        #if lose state, then return negative reward (or maybe 0?)   
        elif(currentState.isLose()):
            return 0
        elif(currentState.isWin()):
            return 1000000 #return some large score 
                
        for i in range(0,5): #keep rolling out unless we hit islose,iswin
            actionList = currentState.getLegalPacmanActions()            
            
            #print("actionList: ", actionList)
            randIdx = random.randint(0,len(actionList)-1)
            randAction = actionList[randIdx]
            currentState = currentState.generatePacmanSuccessor(randAction)
            
            #if none state then we're out of computation resources, break
            if (currentState == None):
                self.ComputationIsNone = True
                return None
            #if lose state, then return negative reward (or maybe 0?)   
            elif(currentState.isLose()):
                return 0
            elif(currentState.isWin()):
                return 1000000 #return some large score 
                
        #now get the score of s compared to bottom of rollout
        #print("s: ", s)
        #print("currentState: ", currentState)
        rewardOfS = gameEvaluation(s,currentState)
        return rewardOfS
    
    def Backup(self,vL,r): #note that r is reward (or the delta) from comparing states b/w begining and end of rollout
        #print("Starting Backup")
        
        while(vL.prev != None):
            #print("vL.prevAction: ", vL.prevAction)
            #print("vL.prev", vL.prev) #seems like it's stuck at the same node
            vL.nj = vL.nj+1
            vL.Q = vL.Q + r #add reward to the total score 
            self.n = self.n+1
            vL = vL.prev
        parentOfV = None
        return parentOfV

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP

        #rand = random.randint(0,-1)
        #print("randint(0,0): ", rand)
        
        #Attempt 2: Doing it with functions
        self.ComputationIsNone = False #not yet out of computation resources
        bestAction = self.MCTSSearch(state)
        #bestAction = Directions.STOP
        return bestAction
        
        
""" Notes

The following Error comes when state.isLose() or state.isWin() is true and you try to get more successors:
if self.isWin() or self.isLose(): raise Exception('Can\'t generate a successor of a terminal state.')
Exception: Can't generate a successor of a terminal state.

"""

