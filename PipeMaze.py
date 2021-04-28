import numpy as np
import pprint
import time

def setUpQ(): # Sets up an empty Q-table with correct values for terminal states.
    states = []
    for y in range(10):
        for x in range(10):
            states.append((x,y))
    actions = [0,1]
    Q = {s: {a: 0 for a in actions} for s in states}
    return Q

def setUpPipe():
    rewardGrid = np.zeros((10,10))
    Q = setUpQ()

    for i in range(10):
        if (i != 6) and (i != 4) and (i != 5):
            rewardGrid[9-i][6] = -1
            rewardGrid[9-i][7] = -1
            Q[(6,i)][0] = -1
            Q[(6,i)][1] = -1
            Q[(7,i)][0] = -1
            Q[(7,i)][1] = -1
        rewardGrid[9-i][9] = 1
        Q[(9,i)][0] = 1
        Q[(9,i)][1] = 1
    return Q, rewardGrid

Q, rewardGrid = setUpPipe()
print(rewardGrid)
# pp.pprint(Q)

def verifyState(x, y):
    valid = True
    if y > 9:
        valid = False
    if y < 0:
        valid = False
    if x > 9:
        valid = False
    return valid

def findPossibleStates(state):
    x, y = state
    states = []
    if verifyState(x+1, y-1):
        states.append((x+1, y-1))
    else:
        if x < 10:
            states.append((x+1, y))
        else: 
            states.append((x,y))
    if verifyState(x+1, y+1):
        states.append((x+1, y+1))
    else:
        if x < 10:
            states.append((x+1, y))
        else: 
            states.append((x,y))
    return states

def chooseEpsilonGreedy(Q, state, epsilon): # Chooses an action based on an epsilon greedy policy.
    p = np.random.rand()
    actionsList = list(Q[state].values())

    if Q[state][0] == 0 and Q[state][1] == 0:
        bestIndex = np.random.randint(2)
    else:
        if p > epsilon:
            bestVal = np.random.choice(actionsList)
            
            bestIndex = actionsList.index(bestVal)
        else:
            bestVal = max(actionsList)
            bestIndex = actionsList.index(bestVal)
            
    return bestIndex

def move(position, action):
    states = findPossibleStates(position)
    return states[action]

def chooseMax(Q, state): # Chooses the action with the highest Q-table value
    # possibleStates = findPossibleStates(state)
    maxVal = Q[state][0]
    maxAction = 0
    if Q[state][1] > maxVal:
        maxAction = 1
    return maxAction

def qLearning(gamma, epsilon, alpha, n):
    np.random.seed(int(time.time()))
    episodes = 0

    Q, rewardGrid = setUpPipe()
    
    s = (0,0)
    x, y = s
    a = chooseEpsilonGreedy(Q, s, epsilon)
    
    for i in range(n):
        if i % 100000 == 0:
            print("Iteration:", i)
        
        # print(s)
        sPrime = move(s,a)
        reward = rewardGrid[9-y,x] #is this right?
        # print(reward)
        
        
        
        if (reward == 1) or (reward == -1): # does not actually ever use the reward, nadav fiiiiix!!
            s = (0,0)
            x, y = s
            a = chooseEpsilonGreedy(Q, s, epsilon)
            episodes += 1
            # print("done")
        else:
            aPrime = chooseMax(Q, sPrime)
            Q[s][a] = Q[s][a] + (alpha * ((reward + gamma*Q[sPrime][aPrime]) - Q[s][a]))
            s = sPrime
            x, y = s
            a = chooseEpsilonGreedy(Q, s, epsilon)

    return Q

gamma = 0.9
epsilon = 0.7
alpha = 0.01
Q = qLearning(gamma, epsilon, alpha, 100000)
pp = pprint.PrettyPrinter(width=41, compact=True)
pp.pprint(Q)
