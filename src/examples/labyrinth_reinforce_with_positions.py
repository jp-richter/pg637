"""
Diese Version nutzt ein sehr einfaches Netzwerk mit nur einer
fully-connected Layer.
Die Information, die wir dem Agenten geben, ist, wo auf welchem
Feld er sich befindet. Er kriegt keine Information darüber, in 
welche Richtung Wände sind oder in welcher Richtung das Ziel liegt.

Um ein bisschen zu veranschaulichen, was passiert habe ich ein paar
Graphen mit der python Bibliothek streamlit gemacht.
Ihr könnt euch das alles auch anzeigen lassen indem ihr
1. Streamlit installiert (pip install streamlit)
2. Dieses python Skript über streamlit ausführt (streamlit run reinforce_with_positions.py)
Dann sollte sich in eurem Browser das streamlit-dashboard öffnen
"""

import torch
import torch.nn as nn
import numpy
import environment
import streamlit

from environment import move, prettyprint


class Policy(nn.Module):
    """
    The policy net consists of two recurrent layers using GRUs, a fully connected layer and a softmax function so that
    the output can be seen as distribution over the possible action choices. The output will be a tensor of size
    (batch size, 1, one hot length) with each index of the one hot dimension representing a action choice.
    """

    def __init__(self):
        super(Policy, self).__init__()

        self.input_dim = 36  # onehot of current position 
        self.output_dim = 4  # action probs
        self.temperature = 1.5

        self.lin = nn.Linear(self.input_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.lin(x)
        out = self.softmax(out / self.temperature)
        return out

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=torch.device(device)))

    def set_parameters_to(self, policy):
        self.load_state_dict(policy.state_dict())


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 16
discount = 0.8
learnrate = 0.02
epochs = 100
simulations = 10
max_steps = 50

directions = [
    'left',
    'right',
    'up',
    'down'
]


def to_onehot(position):
    state = torch.zeros((36,), device=device)
    state[position] = 1
    return state


"""
Macht einen Batch von Vektoren aus einer Liste von Positionen.
Jeder Wert im Vektor steht für eine mögliche Position.
Der Wert v_i ist 1 wenn der Agent gerade an Position i steht und
sonst 0.
"""
def to_batch(positions):
    batch = torch.zeros((batch_size, 36), device=device)

    for i, position in enumerate(positions):
        batch[i] = to_onehot(position)

    return batch


"""
Ein Policy-Gradient Update.
param probs: Ein Tensor [trajectory_length, batch_size] von Logprobabilities der ausgeführten Aktionen
param rewards: Ein Tensor [trajectory_length, batch_size] von Belohnungen, die für die jeweils ausgeführten Aktionen erhalten wurden.
"""
def policy_gradient(optimizer, probs, rewards):
    total = torch.zeros((batch_size,), device=device)
    optimizer.zero_grad()

    # Jeweils ein Schritt für alle Trajektorien im Batch
    for step, (prob, reward) in enumerate(zip(probs, rewards)):
        total = total + discount**step * reward * prob

    total = torch.sum(total) / batch_size

    loss = -total
    loss.backward()
    optimizer.step()



"""
Gegeben ein Batch von jetzigen Zuständen werden Aktionen ausgewählt und ausgeführt.
param policy: Das Policy Netzwerk
param positions: Ein Batch (als Liste) von Feld-IDs. (Nummer des Feldes, wo sich der Agent befindet)
"""
def step(policy, positions):
    batch = to_batch(positions)

    policies = policy(batch)

    # Sample Aktionen (kodiert als Indizes) mit den Wahrscheinlichkeiten der aktuellen Policy
    distributions = torch.distributions.Categorical(policies)
    actions = distributions.sample()
    probs = distributions.log_prob(actions)

    # Transormation der Aktionen in Strings (left, up ...) 
    actions = [directions[index.item()] for index in actions]
    rewards = torch.zeros((batch_size,), device=device)
    next_positions = []

    # Ausführen der Aktionen und Feedback speichern
    for i, (action, position) in enumerate(zip(actions, positions)):
        next_position, reward = move(action, position)

        rewards[i] = reward
        next_positions.append(next_position)

    return next_positions, probs, rewards


"""
Eine Monte-Carlo Simulation, die den Wert eines Zustandes approximieren soll.
param policy: Die Policy der während der Simulation gefolgt werden soll.
param positions: Ein Batch von Positionen (Feld_IDs), für die wir den Wert approximieren wollen.
param simulations: Anzahl der Simulationen, die wir machen. Am Ende wird über alle Simulationen gemittelt.
param steps: Anzahl der Schritte, die wir pro Simulation machen
param current_reward: Die Belohnung für den Schritt der uns in die aktuelle Position gebracht hat.
"""
def montecarlo(policy, positions, simulations, steps, current_reward):
    with torch.no_grad():
        rewards = torch.zeros((simulations, batch_size), device=device)

        for s in range(simulations):
            simulated_rewards = torch.zeros((0, batch_size), device=device)

            for i in range(steps):  # steps
                positions, _, reward = step(policy, positions)
                simulated_rewards = torch.cat((simulated_rewards, reward[None, :]), dim=0)

            rewards[s] = torch.sum(simulated_rewards, dim=0) + current_reward

        rewards = torch.mean(rewards, dim=0)

    return rewards

def favorite_directions(policy):
    inputs = torch.eye(36)
    outputs = policy(inputs) 
    favorite_directions = torch.argmax(outputs, dim=1)
    return favorite_directions

"""
Die eigentliche Trainingsschleife.
"""
def train():
    #STREAMLIT INIT
    my_bar = streamlit.progress(0)
    streamlit.text("Rewards")
    rewards_chart = streamlit.line_chart([0.0])
    streamlit.text("Postions visited in this epoch")
    visit_bar_chart = streamlit.bar_chart()
    streamlit.text("Preferred Actions, " + repr([(i, dir) for i, dir in enumerate(directions)]))
    policy_table = streamlit.table()
    streamlit.text("The Labyrinth: ")
    streamlit.image("../labyrinth.png", width=400)
    #STREAMLIT INIT END

    policy = Policy().to(device)
    rollout = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learnrate)

    for epoch in range(epochs):
        total_reward = 0
        # Wir machen eine Kopie des Netzes für die MC-Sim. Die Kopie heißt Rollout
        rollout.set_parameters_to(policy)
        policy.train()
        # Sehr wichtig in PyTorch, wenn ihr ein Netz nutzt, dass man nicht trainieren will!
        rollout.eval()

        position = [environment.entry_id for _ in range(batch_size)]
        probs = []
        rewards = []

        #DEBUG
        render_positions = {i: [] for i in range(36)}
        #/DEBUG
        
        for current_step in range(max_steps):
            position, prob, reward = step(policy, position)
            total_reward += sum(reward)

            simulation = montecarlo(rollout, position, simulations, 20, reward)

            #DEBUG
            for sample in range(batch_size):
                pos = position[sample]
                val = simulation[sample].item()
                render_positions[pos].append(val)
            #/DEBUG

            rewards.append(simulation)
            probs.append(prob)

        policy_gradient(optimizer, probs, rewards)

        # STREAMLIT OUTPUT 
        my_bar.progress((epoch + 1)/epochs)
        rewards_chart.add_rows([total_reward])
        visit_bar_chart.bar_chart([len(item) for item in render_positions.values()])
        policy_table.table(favorite_directions(rollout).reshape(6, 6).numpy())
        


if __name__ == '__main__':
    train()
