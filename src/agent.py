import random
import torch
from collections import deque
from game import SnakeGameIA
from model import QTrainer, LinearQnet
from visualizer import Visualizer
from copy import deepcopy

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
INPUT_LAYER = 11
HIDDEN_LAYER = 256
SECOND_HIDDEN_LAYER = 256
OUTPUT_LAYER = 3
EPSILON = 100


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON
        self.epsilon_counter = 0
        self.gamma = 0  # This is the rate of discount
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQnet(INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER)
        self.trainer = QTrainer(self.model, LR, GAMMA)

    def get_action(self, state):
        self.epsilon_counter = self.epsilon - self.n_games  # Tradeoff exploration / exploitation
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon_counter:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.forward(state0)
            print("The predict is ", prediction)
            move = torch.argmax(prediction).item()  # Argmax returns the index of the max elements
            # item returns the element in a tensor of 1 x 1 dimensions
            final_move[move] = 1

        return final_move

    def remember(self, state, action, reward, next_states, done):
        self.memory.append((state, action, reward, next_states, done))

    def train_short_memory(self, state, action, reward, next_states, done):
        self.trainer.train_step(state, action, reward, next_states, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Random batch of samples to train
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # * is because it returns to 5 var
        self.trainer.train_step(states, actions, rewards, next_states, dones)


def train():
    plt_scores = []
    plt_mean_scores = []
    plt_total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameIA()
    # new_game = game.copy()
    vis = Visualizer()
    while True:
        state = game.get_state()
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        next_state = game.get_state()
        agent.train_short_memory(state, action, reward, next_state, done)
        agent.remember(state, action, reward, next_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            plt_scores.append(score)
            plt_total_score += score
            mean_score = plt_total_score / agent.n_games
            plt_mean_scores.append(mean_score)
            #  vis.plot(plt_scores, plt_mean_scores, agent.trainer.loss)


if __name__ == "__main__":
    train()
