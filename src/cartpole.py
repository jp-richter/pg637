import gym
import torch
import numpy
import random
import collections


# Num Observation             Min         Max
# 0   Cart Position           -2.4        2.4
# 1   Cart Velocity           -Inf        Inf
# 2   Pole Angle              ~ -41.8°    ~ 41.8°
# 3   Pole Velocity At Tip    -Inf        Inf

env = gym.make('CartPole-v0')

episodes = 10000
steps = 100
learnrate = 0.1
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 16


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4, 64),  # 4d observation vector
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2),  # q values
            torch.nn.LogSoftmax(dim=0) 
        )

    def forward(self, x):
       return self.layers(x)


def train():
    global epsilon

    net = Net().double()
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=learnrate)
    criterion = torch.nn.MSELoss()

    memory = collections.deque(maxlen=1000)  # state, action, next_state, reward

    for episode in range(episodes):
        observation = env.reset()
        done = False

        memory_cache = []  # state, action, next_state

        while not done:
            if numpy.random.random() > epsilon:
                action = torch.argmax(net(torch.from_numpy(observation))).item()

            else:
                action = env.action_space.sample()

            next_observation, _, done, _ = env.step(action)
            memory_cache.append((observation, action, next_observation))

            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            observation = next_observation

        rewards = reversed([numpy.log(i+1) for i in range(len(memory_cache))])
        memory_cache = [(s1, a, s2, r) for ((s1, a, s2), r) in zip(memory_cache, rewards)]
        memory.extend(memory_cache)

        print("Episode finished after {} timesteps".format(rewards))

        if len(memory) < batch_size:
            continue

        batch = random.sample(memory, batch_size)
        batch, _, _, labels = list(zip(*batch))

        batch, labels = torch.tensor(batch), torch.tensor(labels)
        outputs = net(batch)

        # wir brauchen hier noch irgendeine anderen reward für die andere action
        
        loss = criterion(outputs, labels)
        print(loss.item())

train()
