# valueIterationAgents.py
# -----------------------
# THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING ANY
# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR - DANIEL KU.
# ----------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # while still iterations
            # for each state
                # for action in each state
                    # get Q(state, action)
                # store largest (state, action) in Counter. 

        for state in self.mdp.getStates():
            self.values[state] = 0.0

        for i in range(self.iterations):
            next_values = self.values.copy()
            for state in self.mdp.getStates():
                state_values = util.Counter() # store values for actions of this state
                for action in self.mdp.getPossibleActions(state):
                    state_values[action] = self.getQValue(state, action)
                next_values[state] = state_values[state_values.argMax()] # update for each state
            self.values = next_values.copy() # copy new values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # Q(s,a) = Sum_s'( T(s,a,s') + [R(s,a,s') + yV*(s')])
        transition_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        QVal = 0.0
        for transition in transition_probs:
            Tstate, prob = transition
            QVal += prob * (self.mdp.getReward(state, action, Tstate) + self.discount * self.getValue(Tstate))

        return QVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # V*(s) = max_a Q(s,a)
        # Return none if there are no legal actions (aka terminal state)
        if (self.mdp.isTerminal(state)):
            return None
        else:
            QVals = util.Counter()
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                QVals[action] = self.computeQValueFromValues(state, action)

            return QVals.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
