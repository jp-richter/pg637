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

        mean = torch.tanh(self.mean(x)) * 2
        std = torch.tanh(self.std(x)) ** 2

        return mean, std


    def sample(self, x):
        mean, std = self.forward(x)
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        prob = normal.log_prob(action)
        action = torch.tanh(action) * 2

        return action, prob

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file))


env = gym.make('Pendulum-v0')
memory = collections.deque(maxlen=10000)

discount = 0.75
alpha = 0.4
alpha_decay = 0.95
tau = 0.99  # usually close to 1
sigma = 1.0  # influence of memory Q instead of target Q
sigma_decay = 0.95
lr_policy = 0.0003  # 0.0003
lr_q = 0.0003  # 0.0003
episodes = 10000
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
    global alpha, sigma

    memories = random.sample(memory, batchsize)
    states, next_states, actions, rewards, qvalues = list(zip(*memories))

    states = torch.tensor(states)
    next_states = torch.tensor(next_states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    qvalues = torch.tensor(qvalues)

    # update q networks: JQ = 𝔼(s,a)~D[(Q1,2(s,a) - y)^2] 
    # with y = σ𝔼(r)~D[r] + (1 - σ)𝔼(s,a)~D[r + γ(1-d)(min Q1,2(s',a') - αlogπ(a'|s')]

    with torch.no_grad():
        next_actions, next_probs = policy.sample(next_states)
        _, _, minQ = twinQ_target(next_states, next_actions)

    q1, q2, _ = twinQ(next_states, next_actions)
        
    y = rewards + discount * (minQ - alpha * next_probs[:, 0])
    y = sigma * qvalues + (1 - sigma) * y

    q_loss = (crit(q1, y) + crit(q2, y)) / batchsize

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

    sigma *= sigma_decay
    alpha *= alpha_decay


def play(evaluate=False):
    state = env.reset()
    done = False

    with torch.no_grad():
        while not done:
            if not evaluate:
                action, _ = policy.sample(torch.from_numpy(state))

            else:
                action, _ = policy(torch.from_numpy(state))
                env.render()

            next_state, reward, done, _ = env.step(action)
            reward = (reward + 8) / 16

            for step in range(env._elapsed_steps - 1):
                memory[-step][4] += discount ** (step+1) * reward

            memory.append([state, next_state, action, reward, reward])
            state = next_state


def train():
    for episode in range(episodes):
        play()

        if len(memory) > batchsize:
            experience_replay()

        if episode % 1000 == 0:
            print(episode)

    policy.save('/Users/jan/Repositories/pg637/Net_Sigma.net')

    for _ in range(50):
        play(evaluate=True)


train()
