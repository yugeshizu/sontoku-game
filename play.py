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
        policy = agent.get_initial_policy(state)
        reward, state = episode(game, rewards_rec, policy[:, state])
        for _ in range(total_episodes):
            policy = agent.get_updated_policy(reward, state)
            reward, state = episode(game, rewards_rec, policy[:, state])
    else:
        # policy_default: Always choose action=0
        policy_default = np.zeros((len(ACTIONS), len(STATES)))
        policy_default[0, :] = 1
        for _ in range(total_episodes):
            reward, state = episode(game, rewards_rec, policy_default[:, state])

    return rewards_rec


def episode(game, rewards_rec, policy):
    action = np.random.choice(ACTIONS, p=policy)
    reward, state = game.play_an_episode(action)
    rewards_rec.append(reward)
    return reward, state


if __name__ == "__main__":
    rewards_rec = play_a_game_auto(10)
    for i, reward in enumerate(rewards_rec):
        print(
            "Episode={},  reward={},  total_reward={}".format(
                i, reward, sum(rewards_rec[: i + 1])
            )
        )
