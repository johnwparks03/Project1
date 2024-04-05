# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        #Loop through number of iterations
        for i in range (self.iterations):
            new_values = self.values.copy()
            
            #Loop through each state and get q-value for each possible next state
            for state in self.mdp.getStates():
                q_values = []
                #If state is a terminal state set value to 0
                is_terminal_state = self.mdp.isTerminal(state)
                if is_terminal_state:
                    new_values[state] = 0
                else:
                    #Else get maximum q-value from possible actions and set that value to state value
                    possible_actions = self.mdp.getPossibleActions(state)

                    for action in possible_actions:
                        q_values.append(self.getQValue(state, action))
                    
                    new_values[state] = max(q_values)
            self.values = new_values


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
        #Get list of possible next states
        possible_next_states = self.mdp.getTransitionStatesAndProbs(state, action)

        utility_value = 0

        #Calculate utility value for each state and add onto existing value
        for next_state, prob in possible_next_states:
            reward = self.mdp.getReward(state, action, next_state)
            gamma_V = self.discount * self.values[next_state]
            utility_value += (prob * (reward + gamma_V))

        return utility_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possible_actions = self.mdp.getPossibleActions(state)

        #Check if there are no possible actions, if so return None
        if len(possible_actions) == 0:
            return None
        
        #Dctionary to store action and corresponding q-value
        action_qvalues_dict = {}

        #Loop through each action and get the q-value
        for action in possible_actions:
            qValue = self.getQValue(state, action)
            action_qvalues_dict[action] = qValue

        #Return action with highest q-value
        return max(action_qvalues_dict, key= lambda x: action_qvalues_dict[x])



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        #Get mdp states
        states = self.mdp.getStates()

        #Initialize all state values to 0
        for state in states:
            self.values[state] = 0

        for i in range(self.iterations):
            state = states[i % len(states)]
            isTerminal = self.mdp.isTerminal(state)

            #If not in a terminal state get action and corresponding q_value then update the state value
            if not isTerminal:
                action = self.getAction(state)
                q_value = self.getQValue(state, action)
                self.values[state] = q_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        predecessors = {}
        priorityQueue = util.PriorityQueue()
        possible_moves=['north', 'south','east', 'west']

        for state in states:
            self.values[state] = 0
            #Add state predecessors to the dict
            state_predecessors = set()

            if not self.mdp.isTerminal(state):
                for predecessor in states:
                    isTerminal = self.mdp.isTerminal(predecessor)
                    possible_actions = self.mdp.getPossibleActions(predecessor)
                    if not isTerminal:
                        for move in possible_moves:
                            if move in possible_actions:
                                transition = self.mdp.getTransitionStatesAndProbs(predecessor, move)
                                for s_prime, T in transition:
                                    if(s_prime == state) and (T > 0):
                                        state_predecessors.add(predecessor)
            predecessors[state] = state_predecessors


        for s in states:
            isTerminal = self.mdp.isTerminal(s)
            #If state is not terminal calculate diff between maximum q-value and current value
            if not isTerminal:
                current_state_value = self.values[s]
                diff = abs(current_state_value - self.get_max_q_value(s))
                #Push to the priority queue with -diff because of min-heap
                priorityQueue.push(s, -diff)

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                return
            state = priorityQueue.pop()
            self.values[state] = self.get_max_q_value(state)

            for predecessor in predecessors[state]:
                predecessor_max_q_value = self.get_max_q_value(predecessor)
                diff = abs(self.values[predecessor] - predecessor_max_q_value)
                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)
    
    def get_max_q_value(self, state):
        return max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
