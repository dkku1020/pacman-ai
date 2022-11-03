# THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING ANY
# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR - DANIEL KU.
# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        Food = newFood.asList()
        minfoodDist = float("inf")
        ghostDist = []
        
        # Calculate distances from food to newPos
        for food in Food:
          minfoodDist = min(minfoodDist, manhattanDistance(food, newPos))
        # Calculate distances from Ghost to newPos
        for ghost in successorGameState.getGhostPositions():
          ghostDist.append(manhattanDistance(ghost, newPos))
      
        # To prevent pacman from getting too close to Ghosts
        for dist in ghostDist:
          if dist < 2:
            return -(float("inf"))

        # To prevent pacman from not moving
        if currentGameState.getPacmanPosition() == newPos:
          return -(float("inf"))
        
        ## Basic Eval function makes pacman choose closest food while avoiding Ghosts
        return successorGameState.getScore() + 1.0/minfoodDist 

        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.miniMax(gameState, 0, 0)[0]

    def minValue(self, gameState, depth, agentIndex):
      minV = ["", float("inf")]
      ghostActions = gameState.getLegalActions(agentIndex)

      if not ghostActions:
        return self.evaluationFunction(gameState)
          
      for action in ghostActions:
        currentState = gameState.generateSuccessor(agentIndex, action)
        currentV = self.miniMax(currentState, depth, agentIndex + 1)
        if type(currentV) is not list:
          newValue = currentV
        else:
          newValue = currentV[1]
        if newValue < minV[1]:
          minV = [action, newValue]
      return minV

    def maxValue(self, gameState, depth, agentIndex):
      maxV = ["", -float("inf")]
      pacmanActions = gameState.getLegalActions(agentIndex)

      if not pacmanActions:
        return self.evaluationFunction(gameState)

      for action in pacmanActions:
        currentState = gameState.generateSuccessor(agentIndex, action)
        currentV = self.miniMax(currentState, depth, agentIndex + 1)
        if type(currentV) is not list:
          newValue = currentV
        else:
          newValue = currentV[1]
        if newValue > maxV[1]:
          maxV = [action, newValue]
      return maxV

    def miniMax(self, gameState, depth, agentIndex):
      ## Reset agentIndex every ply and increase depth
      if agentIndex >= gameState.getNumAgents():   
        depth += 1
        agentIndex = 0
          
      if (depth == self.depth or gameState.isWin() or gameState.isLose()):
        return self.evaluationFunction(gameState)
      ## maxValue if pacman (agent 0)
      elif (agentIndex == 0):
        return self.maxValue(gameState, depth, agentIndex) 
      ## minValue if ghost (! agent 0)
      else:
        return self.minValue(gameState, depth, agentIndex)

        
        





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.miniMax(gameState, 0, 0, -float("Inf"), float("Inf"))[0]

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
      minV = ["", float("inf")]
      ghostActions = gameState.getLegalActions(agentIndex)

      if not ghostActions:
        return self.evaluationFunction(gameState)
          
      for action in ghostActions:
        currentState = gameState.generateSuccessor(agentIndex, action)
        currentV = self.miniMax(currentState, depth, agentIndex + 1, alpha, beta)
        if type(currentV) is not list:
          newValue = currentV
        else:
          newValue = currentV[1]
        if newValue < minV[1]:
          minV = [action, newValue]

        # Prunning
        if newValue < alpha:
          return [action, newValue]
        beta = min(beta, newValue)

      return minV

    def maxValue(self, gameState, depth, agentIndex, alpha, beta):
      maxV = ["", -float("inf")]
      pacmanActions = gameState.getLegalActions(agentIndex)

      if not pacmanActions:
        return self.evaluationFunction(gameState)

      for action in pacmanActions:
        currentState = gameState.generateSuccessor(agentIndex, action)
        currentV = self.miniMax(currentState, depth, agentIndex + 1, alpha, beta)
        if type(currentV) is not list:
          newValue = currentV
        else:
          newValue = currentV[1]
        if newValue > maxV[1]:
          maxV = [action, newValue]

        # Prunning
        if newValue > beta:
          return [action, newValue]
        alpha = max(alpha, newValue)

      return maxV

    def miniMax(self, gameState, depth, agentIndex, alpha, beta):
      ## Reset agentIndex every ply and increase depth
      if agentIndex >= gameState.getNumAgents():   
        depth += 1
        agentIndex = 0
          
      if (depth == self.depth or gameState.isWin() or gameState.isLose()):
        return self.evaluationFunction(gameState)
      ## maxValue if pacman (agent 0)
      elif (agentIndex == 0):
        return self.maxValue(gameState, depth, agentIndex, alpha, beta) 
      ## minValue if ghost (! agent 0)
      else:
        return self.minValue(gameState, depth, agentIndex, alpha, beta) 

        
        


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectiMaxValue(gameState, 0, 0)[0]

    def probValue(self, gameState, depth, agentIndex):
      expMax = ["", 0]
      ghostActions = gameState.getLegalActions(agentIndex)
      probability = 1.0 / len(ghostActions)

      if not ghostActions:
        return self.evaluationFunction(gameState)
          
      for action in ghostActions:
        currentState = gameState.generateSuccessor(agentIndex, action)
        currentV = self.expectiMaxValue(currentState, depth, agentIndex + 1)
        if type(currentV) == list:
          newValue = currentV[1]
        else:
          newValue = currentV
        expMax[0] = action
        expMax[1] += newValue * probability
      return expMax

    def maxValue(self, gameState, depth, agentIndex):
      maximum = ["", -float("Inf")]
      pacmanActions = gameState.getLegalActions(agentIndex)

      if not pacmanActions:
        return self.evaluationFunction(gameState)

      for action in pacmanActions:
        currentState = gameState.generateSuccessor(agentIndex, action)
        currentV = self.expectiMaxValue(currentState, depth, agentIndex + 1)
        if type(currentV) is not list:
          newValue = currentV
        else:
          newValue = currentV[1]
        if newValue > maximum[1]:
          maximum = [action, newValue]
      return maximum

    def expectiMaxValue(self, gameState, depth, agentIndex):
      ## Reset agentIndex every ply and increase depth
      if agentIndex >= gameState.getNumAgents():  
        depth += 1
        agentIndex = 0

      if (depth == self.depth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
      # if pacman (MAX agent): return max successor value
      elif (agentIndex == 0):
        return self.maxValue(gameState, depth, agentIndex)
      # if Ghost (EXP agent) - return probability Value
      else:
        return self.probValue(gameState, depth, agentIndex)

        
       
              

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I made my betterEvaluationFunction with the idea of making pacman eat capsules available
      and take out ghosts nearby if possible.
      The main objective is still to eat all the food pallets (decreasing number of foods left on map)
      while taking advantage of capsules on the map and scaring the ghosts.
      I devised a scoring system that encourages pacman to eat capsules that are near
      and eat ghosts that are scared if they are nearby.

      Some factors that I assigned were:
      1) Food distance from currentPos
      2) Number of Food Pallets left
      3) A scoring system that makes pacman want to eat capsules and then eat Ghosts


    """
    "*** YOUR CODE HERE ***"
    # initialize currentPos, currentFood, capsulePos, and layout of Map
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    capsulePos = currentGameState.getCapsules()
    layout = currentGameState.getWalls()

    foodDist = []
    capsuleDist = []

    # Take manhattan distance to food and capsule from currentPos and append to list
    for food in currentFood.asList():
      foodDist.append(manhattanDistance(currentPos, food))
    for capsule in capsulePos:
      capsuleDist.append(manhattanDistance(currentPos, capsule))

    # initialize score to 0
    score = 0
    # take pacmans x and y coordinates
    x = currentPos[0]
    y = currentPos[1]

    # Take manhattan distance to ghosts from currentPos
    for ghostState in currentGameState.getGhostStates():
      ghostDist = manhattanDistance(currentPos, ghostState.configuration.getPosition())
      # Check if ghostDist < 2
      if ghostDist < 2:
        # Check if we can eat Ghost
        if ghostState.scaredTimer != 0:
          # If we can eat, make pacman eat ghost
          score += 1000.0/(ghostDist+1)
        else:
          # If NOT, make pacman get away
          score -= 1000.0/(ghostDist+1)

    # Check if capsule is within 5
    # Add float(100) incase there are no capsules on the map
    if min(capsuleDist + [float(100)]) < 5:
      score += 500.0/(min(capsuleDist))

    # The code above makes pacman get closer to capsule but does not make pacman eat capsule
    # Check if pacman can eat capsule, and make pacman eat capsule if possible by assigning
    # higher score (800 > 500)
    for capsule in capsulePos:
      if (capsule[0]==x) & (capsule[1]==y):
        score += 800.0

    # Initialize minFoodDist
    # Add float(100) incase there are no food left on the map
    minFoodDist = min(foodDist+[float(100)])

    # Final Evaluation Function
    return score + 1.0/minFoodDist - len(foodDist)*10.0

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

