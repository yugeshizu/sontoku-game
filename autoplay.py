import numpy as np
from sontoku import Sontoku


STATES = Sontoku.STATES
ACTIONS = Sontoku.ACTIONS


def play_a_game_auto(total_episodes=10, agent=None):
    game = Sontoku()
    game.initialize_state()
    state = game.state
    rewards_rec = [0]
    if agent is not None:
        agent.initialize(state)
        for _ in range(total_episodes):
            action = agent.get_next_action()
            reward, state = game.play_an_episode(action)
            rewards_rec.append(reward)
            agent.update(reward, state)
    else:
        # policy_default: Always choose action=0
        action = 0
        for _ in range(total_episodes):
            reward, state = game.play_an_episode(action)
            rewards_rec.append(reward)

    return rewards_rec


if __name__ == "__main__":
    rewards_rec = play_a_game_auto(10)
    for i, reward in enumerate(rewards_rec):
        print(
            "Episode={},  reward={},  total_reward={}".format(
                i, reward, sum(rewards_rec[: i + 1])
            )
        )
