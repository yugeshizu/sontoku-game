import numpy as np
from sontoku import Sontoku


STATES = Sontoku.STATES
ACTIONS = Sontoku.ACTIONS


class Agent:
    def __init__(self):
        # policy_default: Always choose action=0
        policy_default = np.zeros((len(ACTIONS), len(STATES)))
        policy_default[0, :] = 1
        self.policy = policy_default

    def get_initial_policy(self, state=STATES[0]):
        return self.policy

    def get_updated_policy(self, reward=0, next_state=STATES[0]):
        return self.policy


if __name__ == "__main__":
    from autoplay import play_a_game_auto

    agent = Agent()
    rewards_rec = play_a_game_auto(10, agent)
    for i, reward in enumerate(rewards_rec):
        print(
            "Episode={},  reward={},  total_reward={}".format(
                i, reward, sum(rewards_rec[: i + 1])
            )
        )
