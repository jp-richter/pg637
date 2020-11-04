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
            torch.nn.Linear(64, 2)  # q values
        )

    def forward(self, x):
       return self.layers(x)


def train():
    global epsilon

    net = Net().double()
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=learnrate)
    criterion = torch.nn.MSELoss()

    memory = collections.deque(maxlen=1000)  # state, action, next_state, done, reward

    for episode in range(episodes):
        state = env.reset()
        net.eval()

        memory_cache = []  # state, action, next_state, done
        done = False

        # Compute a single trajectory

        while not done:
            if numpy.random.random() > epsilon:
                action = torch.argmax(net(torch.from_numpy(state))).item()

            else:
                action = env.action_space.sample()

            next_state, _, done, _ = env.step(action)
            memory_cache.append((state, action, next_state, done))

            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            state = next_state

        rewards = reversed([numpy.log(i+1) for i in range(len(memory_cache))])  # log n, log n-1, ..
        memory_cache = list(zip(*zip(*memory_cache), rewards))  # state, action, next_state, done, reward
        memory.extend(memory_cache)

        print("Episode finished after {} timesteps".format(len(memory_cache)))

        # Experience replay on memorized states

        net.train()

        if len(memory) < batch_size:
            continue

        memories = random.sample(memory, batch_size)
        states, actions, next_states, dones, rewards = list(zip(*memories))
        states, next_states, rewards = torch.tensor(states), torch.tensor(next_states), torch.tensor(rewards)

        predictions = net(states)

        qvalues = rewards + 0.9 * torch.amax(net(next_states), dim=1)
        targets = predictions.detach().clone()

        for i in range(batch_size):
            if not dones[i]:
                targets[i][actions[i]] = qvalues[i]

            else:
                targets[i][actions[i]] = -1
        
        optimizer.zero_grad()
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

train()
