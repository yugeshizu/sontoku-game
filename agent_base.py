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

    def initialize(self, state=STATES[0]):
        pass

    def get_next_action():
        return 0

    def updated_policy(self, reward=0, next_state=STATES[0]):
        pass


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
