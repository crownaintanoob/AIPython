import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
RaycastsAmount = round(360 / 10)
InputsAmount = 5
rotationAxis = 3
OtherStates = 1

class Agent:

    def __init__(self):
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(OtherStates + rotationAxis + RaycastsAmount, 128, InputsAmount) # [input_size, hidden_size, output_size] (8, 128, 3)
        self.model.load_state_dict(torch.load(r'C:\Users\liong\Desktop\synapse_x\AIPythonSwordFighting\models\model.pth'))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, stateArray):
        return np.array(stateArray, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80
        final_move = []
        MinValueForPrediction = 2
        for _ in range(InputsAmount):
            final_move.append(0)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, InputsAmount - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            for i, x in enumerate(prediction):
                if x.item() >= MinValueForPrediction:
                    final_move[i] = 1
            #move = torch.argmax(prediction).item()
            #final_move[move] = 1

        return final_move