import numpy as np
import pprint
import time

height = 20
length = 20

def setUpQ(): # Sets up an empty Q-table with correct values for terminal states.
    global height, length
    states = []
    for y in range(height):
        for x in range(length):
            states.append((x,y))
    actions = [0,1]
    Q = {s: {a: 0 for a in actions} for s in states}
    return Q

def setUpOnePipe(rewardGrid, x, y):
    global height, length
    for i in range(height):
        if (i != y) and (i != y+1) and (i != y+2):
            rewardGrid[(height-1)-i][x] = -1
            rewardGrid[(height-1)-i][x+1] = -1

def setUpPipes():
    global height, length
    rewardGrid = np.zeros((height,length))
    Q = setUpQ()

    setUpOnePipe(rewardGrid, 6, 4)
    setUpOnePipe(rewardGrid, 14, 8)
    for i in range(height):
        # if (i != 6) and (i != 4) and (i != 5):
        #     rewardGrid[(height-1)-i][6] = -1
        #     rewardGrid[(height-1)-i][7] = -1
        #     # Q[(6,i)][0] = -1
        #     # Q[(6,i)][1] = -1
        #     # Q[(7,i)][0] = -1
        #     # Q[(7,i)][1] = -1
        rewardGrid[(height-1)-i][length-1] = 1
        # Q[(9,i)][0] = 1
        # Q[(9,i)][1] = 1
    return Q, rewardGrid

Q, rewardGrid = setUpPipes()
print(rewardGrid)
# pp.pprint(Q)

def verifyState(x, y):
    global height, length
    valid = True
    if y > height -1:
        valid = False
    if y < 0:
        valid = False
    if x > length - 1:
        valid = False
    return valid

def findPossibleStates(state):
    global height, length
    x, y = state
    states = []
    if verifyState(x+1, y-1):
        states.append((x+1, y-1))
    else:
        if x < length:
            states.append((x+1, y))
        else: 
            states.append((x,y))
    if verifyState(x+1, y+1):
        states.append((x+1, y+1))
    else:
        if x < length:
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
    global length, height
    np.random.seed(int(time.time()))
    episodes = 0

    Q, rewardGrid = setUpPipes()
    
    s = (0,0)
    x, y = s
    a = chooseEpsilonGreedy(Q, s, epsilon)
    
    for i in range(n):
        if i % 100000 == 0:
            print("Iteration:", i)
        
        # print(s)
        sPrime = move(s,a)
        x1, y1 = sPrime
        reward = rewardGrid[(height-1)-y1,x1] #is this right?
        # print(reward)
        
        
        aPrime = chooseMax(Q, sPrime)
        Q[s][a] = Q[s][a] + (alpha * ((reward + gamma*Q[sPrime][aPrime]) - Q[s][a]))

        if (reward == 1) or (reward == -1): # does not actually ever use the reward, nadav fiiiiix!!
            s = (0,0)
            x, y = s
            a = chooseEpsilonGreedy(Q, s, epsilon)
            episodes += 1
            # print("done")
        else:
            s = sPrime
            x, y = s
            a = chooseEpsilonGreedy(Q, s, epsilon)

    return Q

gamma = 0.9
epsilon = 0.7
alpha = 0.1
Q = qLearning(gamma, epsilon, alpha, 1000000)
pp = pprint.PrettyPrinter(width=41, compact=True)
# pp.pprint(Q)


for x in range(length):
    print(x, "\t", end= "")
print()
for y in reversed(range(height)):
    for x in range(length):
        maxVal = max(Q[(x,y)][0], Q[(x,y)][1])
        print(round(maxVal, 2), "\t", end= "")
    print()

print("\n")
for x in range(length):
    print(x, "\t", end= "")
print()
for y in reversed(range(height)):
    for x in range(length):
        if rewardGrid[(height-1)-y][x] == -1 or rewardGrid[(height-1)-y][x] == 1:
            print(rewardGrid[(height-1)-y][x], "\t", end= "")
        elif (Q[(x,y)][0] < Q[(x,y)][1]):
            print("jump", "\t", end= "")
        else:
            print("fall", "\t", end= "")
        
    print()