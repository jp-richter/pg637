import gym
import torch
import torch.nn
import torch.distributions
import collections
import random


class TwinQ(torch.nn.Module):

    def __init__(self):
        super(TwinQ, self).__init__()
        
        self.q1 = torch.nn.Sequential(
            torch.nn.Linear(4, 128),  # (observation + action) ->..
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)  # ..-> (q value)
        )

        self.q2 = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)

        q1 = self.q1(x)[:, 0]
        q2 = self.q2(x)[:, 0]

        return q1, q2, torch.min(q1, q2)

    def update(self, policy, tau=1.0):
        for p1, p2 in zip(self.parameters(), policy.parameters()):
            p1.data.copy_(p1.data * (1.0 - tau) + p2.data * tau)


class Policy(torch.nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 128),  # (observation) ->..
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU()
        )

        self.mean = torch.nn.Linear(128, 1)
        self.std = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.layers(x)

        return self.mean(x), self.std(x) ** 2  # enforce positive std


    def sample(self, x):
        mean, std = self.forward(x)
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        prob = normal.log_prob(action)
        action = torch.tanh(action)

        return action, prob


env = gym.make('Pendulum-v0')
memory = collections.deque(maxlen=10000)

discount = 0.99
alpha = 0.2
tau = 0.95  # usually close to 1
lr_policy = 0.01  # 0.0003
lr_q = 0.01  # 0.0003
episodes = 1000
batchsize = 16

policy = Policy().double()
twinQ = TwinQ().double()
twinQ_target = TwinQ().double()

policy.train()
twinQ.train()
twinQ_target.update(twinQ)

opt_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy)
opt_q = torch.optim.Adam(twinQ.parameters(), lr=lr_q)
crit = torch.nn.MSELoss()


def experience_replay():
    memories = random.sample(memory, batchsize)
    states, next_states, actions, rewards, mask =  list(zip(*memories))

    states = torch.tensor(states)
    next_states = torch.tensor(next_states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    mask = torch.tensor(mask)

    # update q networks: JQ = 𝔼(s,a)~D[(Q1,2(s,a) - y)^2] with y = r + γ(1-d)(min Q1,2(s',a') - αlogπ(a'|s'))

    with torch.no_grad():
        next_actions, next_probs = policy.sample(next_states)
        _, _, minQ = twinQ_target(next_states, next_actions)
        y = rewards + discount * mask * (minQ - alpha * next_probs[:, 0])

    q1, q2, _ = twinQ(next_states, next_actions)

    q1_loss = crit(q1, y)
    q2_loss = crit(q2, y)
    q_loss = (q1_loss + q2_loss) / batchsize

    opt_q.zero_grad()
    q_loss.backward()
    opt_q.step()

    # update policy network: Jπ = - 𝔼s∼D,ε∼N[Q(s,f(ε,s)) - αlogπ(f(ε,s)|s)]

    next_actions, next_probs = policy.sample(next_states)

    _, _, minQ = twinQ(next_states, next_actions)
    policy_loss = -(minQ - alpha * next_probs).mean()  # 𝔼(αH(π)) = 𝔼(-αlogπ)! :O

    opt_policy.zero_grad()
    policy_loss.backward()
    opt_policy.step()

    # polyak average the Q-network parameters over course of training to obtain targets

    twinQ_target.update(twinQ, tau)

    return q_loss.item(), policy_loss.item()


def train():
    rewards = []
    q_losses = []
    p_losses = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        rew = 0

        with torch.no_grad():
            while not done:
                action, _ = policy.sample(torch.from_numpy(state))
                next_state, reward, done, _ = env.step(action)
                mask = 1.0 if env._elapsed_steps == env._max_episode_steps else float(not done)
                memory.append((state, next_state, action, reward, mask))

                rew += reward

        rewards.append(rew)
        rew = 0

        if len(memory) > batchsize:
            qloss, ploss = experience_replay()
            q_losses.append(qloss)
            p_losses.append(ploss)


    import matplotlib.pyplot

    x = [i for i in range(len(rewards))]

    matplotlib.pyplot.plot(x, rewards)
    matplotlib.pyplot.show()

    matplotlib.pyplot.plot(x, q_losses)
    matplotlib.pyplot.show()

    matplotlib.pyplot.plot(x, p_losses)
    matplotlib.pyplot.show()


train()
