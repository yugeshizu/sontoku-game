import numpy as np
from agent_base import Agent
import policy_models
from sontoku import Sontoku


STATES = Sontoku.STATES
ACTIONS = Sontoku.ACTIONS
REWARD_MAX = 10

model_default = policy_models.policy_model_default


class AgentSARSA(Agent):
    def __init__(self, gamma=0.9, policy_model=model_default, optimistic=False):
        super().__init__()
        self.gamma = gamma
        self.episode_counts = 0
        self.policy_model = policy_model
        if optimistic:
            self.q_func = np.full((len(STATES), len(ACTIONS)), REWARD_MAX / (1 - gamma))
        else:
            self.q_func = np.zeros((len(STATES), len(ACTIONS)))

    def initialize(self, state):
        self.policy = self.policy_default
        self.state = state
        self.action = np.random.choice(ACTIONS, p=self.policy[:, state])

    def get_next_action():
        return self.action

    def update(self, reward=0, next_state=STATES[0]):
        next_action = np.random.choice(ACTIONS, p=self.policy[:, next_state])
        self.update_q_func(reward, next_state, next_action)
        self.update_policy()
        self.state = next_state
        self.action = next_action
        self.episode_counts += 1

    def update_q_func(self, reward, next_state, next_action):
        error_TD = (
            reward
            + self.gamma * self.q_func[next_state, next_action]
            - self.q_func[self.state, self.action]
        )
        learning_rate = 100 / (self.episode_counts + 200)
        self.q_func[self.state, self.action] += learning_rate * error_TD

    def update_policy(self):
        param = 100 / (self.episode_counts + 200)
        self.policy[:, self.state] = self.policy_model(self.state, self.q_func, param)


if __name__ == "__main__":
    from autoplay import play_a_game_auto

    agent = AgentSARSA()
    rewards_rec = play_a_game_auto(10, agent)
    for i, reward in enumerate(rewards_rec):
        print(
            "Episode={},  reward={},  total_reward={}".format(
                i, reward, sum(rewards_rec[: i + 1])
            )
        )
