import numpy as np


class Sontoku:
    # 取りうる状態：テキストから番号を1だけずらした（減らした）
    STATES = (0, 1, 2)
    # 取りうる行動：テキストから番号を1だけずらした（減らした）
    ACTIONS = (0, 1)

    # 遷移確率：transition_probs[遷移後の状態, 現在の状態, 行動]
    transition_probs = np.zeros((len(STATES), len(STATES), len(ACTIONS)))
    transition_probs[0, 0, 0] = 1.0
    transition_probs[0, 0, 1] = 0.5
    transition_probs[0, 1, 0] = 1.0
    transition_probs[0, 1, 1] = 0.0
    transition_probs[0, 2, 0] = 0.0
    transition_probs[0, 2, 1] = 1.0
    transition_probs[1, 0, 1] = 0.5
    transition_probs[1, 1, 0] = 0.0
    transition_probs[1, 1, 1] = 0.5
    transition_probs[1, 2, 0] = 1.0
    transition_probs[1, 2, 1] = 0.0
    transition_probs[2, 0, 0] = 0.0
    transition_probs[2, 0, 1] = 0.0
    transition_probs[2, 1, 0] = 0.0
    transition_probs[2, 1, 1] = 0.5
    transition_probs[2, 2, 0] = 0.0
    transition_probs[2, 2, 1] = 0.0

    # 報酬：rewards[現在の状態, 行動]
    rewards = np.zeros((len(STATES), len(ACTIONS)), dtype=np.int32)
    rewards[0, 0] = 1  # テキストではここの報酬が0だが1の誤植だと思われる
    rewards[0, 1] = -1
    rewards[1, 0] = 1
    rewards[1, 1] = -1
    rewards[2, 0] = 1
    rewards[2, 1] = 12

    def __init__(self):
        self.initialize_state()

    def get_reward(self, action):
        return self.rewards[self.state, action]

    def initialize_state(self):
        self.state = np.random.choice(self.STATES)

    def update_state(self, action):
        self.state = np.random.choice(
            self.STATES, p=self.transition_probs[:, self.state, action]
        )

    def play_an_episode(self, action):
        reward = self.get_reward(action)
        self.update_state(action)
        return reward, self.state

    def play_a_game_with_keyboard(self, total_episodes=20):
        self.initialize_state()
        total_reward = 0
        episode_counts = 0
        print("--------")
        print("Start.")
        print("Total reward = {}".format(total_reward))
        print("State = {}".format(self.state))
        print("--------")
        while episode_counts < total_episodes:
            print("Set action: 0 or 1.  Finish: other keys.")
            action = input("Press a key: ")
            if not (action == "0" or action == "1"):
                break
            episode_counts += 1
            reward, next_state = self.play_an_episode(int(action))
            total_reward += reward
            print("--------")
            print("Played {} rounds.".format(episode_counts))
            print("Reward = {}".format(reward))
            print("Total reward = {}".format(total_reward))
            print("State = {}".format(next_state))
            print("--------")

        print("--------")
        print("Finish.")
        print("Played {} rounds.".format(episode_counts))
        print("Total reward = {}".format(total_reward))


if __name__ == "__main__":
    g = Sontoku()
    g.play_a_game_with_keyboard()
