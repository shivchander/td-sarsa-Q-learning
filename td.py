import numpy as np
import random
from tqdm import tqdm

gamma = 0.98
gridSize = 8
alpha = 0.9
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

# boundary states
top_states = [[0, i] for i in range(1, 7)]
bot_states = [[7, i] for i in range(1, 7)]
left_states = [[i, 0] for i in range(1, 7)]
right_states = [[i, 7] for i in range(1, 7)]
start_state = [[7, 0]]
top_right_state = [[0, 7]]
bot_right_state = [[7, 7]]
winning_states = [[1, 0], [0, 1]]
terminationState = [0, 0]

# set of actions
action_key = {'L': [0, -1], 'R': [0, 1], 'U': [-1, 0], 'D': [1, 0], 'Exit': [0, 0]}
start_actions = ['R', 'U']
general_actions = ['L', 'R', 'U', 'D']
right_actions = ['L', 'U', 'D']
left_actions = ['R', 'U', 'D']
top_actions = ['D', 'L', 'R']
bot_actions = ['U', 'L', 'R']
top_right_actions = ['L', 'D']
bot_right_actions = ['L', 'U']
terminal_action = ['Exit']
numIterations = 1000

# ladders
ladder_1 = [[7, 5], [4, 2]]
ladder_2 = [[3, 3], [0, 0]]

# snake
snake = [[0, 2], [3, 6]]


def policy(initialPosition, action, reward_R):
    if initialPosition == terminationState:
        return initialPosition, 0
    if initialPosition == snake[0]:
        return snake[1], -3
    if initialPosition == ladder_1[0]:
        return ladder_1[1], reward_R
    if initialPosition == ladder_2[0]:
        return ladder_2[1], 15

    finalPosition = list(np.array(initialPosition) + np.array(action_key[action]))
    if finalPosition == snake[0]:
        return snake[1], -3
    elif finalPosition == ladder_1[0]:
        return ladder_1[1], reward_R
    elif finalPosition == ladder_2[0]:
        return ladder_2[1], 15
    elif finalPosition == terminationState:
        return terminationState, 9.5
    elif -1 in finalPosition or 8 in finalPosition:
        finalPosition = initialPosition
        return finalPosition, 0
    else:
        return finalPosition, -0.5


def policy_eval(R):
    V = np.zeros((gridSize, gridSize))
    deltas = {(i, j): list() for i in range(gridSize) for j in range(gridSize)}

    for it in tqdm(range(int(1e4))):
        state = random.choice(states[1:-1])
        while True:
            if state in start_state:
                actions = start_actions
            elif state in top_states:
                actions = top_actions
            elif state in bot_states:
                actions = bot_actions
            elif state in left_states:
                actions = left_actions
            elif state in right_states:
                actions = right_actions
            elif state in top_right_state:
                actions = top_right_actions
            elif state in bot_right_state:
                actions = bot_right_actions
            elif state == terminationState:
                actions = terminal_action
            else:
                actions = general_actions
            action = random.choice(actions)
            finalState, reward = policy(state, action, reward_R=R)

            # we reached the end
            if finalState == terminationState:
                break

            # modify Value function
            before = V[state[0], state[1]]
            V[state[0], state[1]] += alpha * (reward + gamma * V[finalState[0], finalState[1]] - V[state[0], state[1]])
            deltas[state[0], state[1]].append(float(np.abs(before - V[state[0], state[1]])))

            state = finalState

    return V
