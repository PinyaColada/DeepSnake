import torch
import numpy as np
import torch.nn.functional as fu
import torch.nn as nn
import torch.optim as optim
import os


class LinearQnet(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super().__init__()
        self.lineal1 = nn.Linear(input_layer, hidden_layer)
        self.lineal2 = nn.Linear(hidden_layer, output_layer)

    def forward(self, x):
        x = fu.relu(self.lineal1(x))
        x = self.lineal2(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)  # makedirs creates a path

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss = []

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # We do this because a 0 dimensional tensor can't be accessed with aux[0] and it must be done with .item()
            # Making the code different if it has been called by short memory or long memory
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        q = self.model.forward(state)  # Generate our quality from the action
        q_new = q.clone()

        for i in range(len(state)):
            predict = reward[i]  # we get the reward for the actions that were made
            if not done[i]:
                # We calculate with the following state quality
                predict += self.gamma * torch.max(self.model.forward(next_state[i]))

            q_new[i][torch.argmax(action[i]).item()] = predict

        self.optimizer.zero_grad()
        loss = self.criterion(q, q_new)
        loss.backward()

        self.optimizer.step()
