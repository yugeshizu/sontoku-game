import numpy as np
from sontoku import Sontoku


STATES = Sontoku.STATES
ACTIONS = Sontoku.ACTIONS

# policy_default: Always choose action=0
policy_default = np.zeros((len(ACTIONS), len(STATES)))
policy_default[0, :] = 1

policy_default_1 = np.zeros((len(ACTIONS), len(STATES)))
policy_default_1[1, :] = 1

policy_random = np.full((len(ACTIONS), len(STATES)), 1 / len(ACTIONS))


def policy_model_default(state, q_func=None, param=None):
    return policy_default[:, state]


def policy_model_default_1(state, q_func=None, param=None):
    return policy_default_1[:, state]


def epsilon_greedy(state, q_func, epsilon=0.3):
    if np.random.rand() < epsilon:
        return policy_random[:, state]
    else:
        policy_at_state = np.zeros(len(ACTIONS))
        policy_at_state[np.argmax(q_func[state, :])] = 1
        return policy_at_state


def softmax(state, q_func, temperature=0.3):
    beta = 1 / temperature
    q_max = np.max(q_func[state, :])
    policy_at_state = np.exp(beta * (q_func[state, :] - q_max))
    normalization_factor = np.sum(policy_at_state)
    return policy_at_state / normalization_factor
