import torch
import numpy
import environment as env

from environment import get_valid_directions, move, prettyprint


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.input_dim = 4  # onehot of possible paths
        self.output_dim = 4  # action probs
        self.hidden_dim = 32
        self.layers = 2
        self.temperature = 1.2

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.layers, batch_first=True)
        self.lin = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h=None):
        if h is not None:
            out, h = self.gru(x, h)

        else:
            out, h = self.gru(x)

        out = out[:, -1]
        out = self.lin(out)
        out = self.relu(out)
        out = self.softmax(out / self.temperature)

        return out, h

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
with_baseline = True

# An dieser Stelle nicht genannt und etwas mit dem man rumspielen kann ist ein temperature Parameter im GRU Netz. Der smoothed ein
# wenig die Policy und soll verhindern, dass die Funktion nicht auf ungewünschte Modalwerte kollabiert. Das kann gerade am Anfang schnell
# passieren.

# Außerdem wird im environment ein negativer Reward von 0.5 für das gegen die Wand laufen gegeben.

# Der Prozess hängt extrem stark vom Zufall ab! Es kann durchaus Runs geben, bei denen mit den gegebenen Epochen und Parametern kein 
# nennenswerter Erfolg erzielt wird. Es sollte aber abgesehen von Ausreißern recht zuverlässig funktionieren. Man muss das zum Testen
# auch nicht komplett durchlaufen lassen. 


direction_indices = {
    'left': 0,
    'right': 1,
    'up': 2,
    'down': 3
}

direction_strings = {
    v: k for (k, v) in direction_indices.items()
}


def to_onehot(directions):
    state = torch.zeros((4,), device=device)

    for direction in directions:
        state[direction_indices[direction]] = 1

    return state


"""
Gibt eine Matrix der Größe (batchsize, 1, 4) zurück, wobei jedes Element in Dimension 0 ein one hot kodierter Zustand ist. Die 1 in
der Mitte kann man ignorieren. Das ist die Eingabe in das Netz.
"""
def to_batch(directions):
    batch = torch.zeros((batch_size, 1, 4), device=device)

    for i, dirs in enumerate(directions):
        batch[i] = to_onehot(dirs)

    return batch


cache = []


"""
Ein Policy-Gradient Update.
param probs: Ein Tensor [trajectory_length, batch_size] von Logprobabilities der ausgeführten Aktionen
param rewards: Ein Tensor [trajectory_length, batch_size] von Belohnungen, die für die jeweils ausgeführten Aktionen erhalten wurden.
"""
def policy_gradient(optimizer, probs, rewards):
    total = torch.zeros((batch_size,), device=device)
    optimizer.zero_grad()

    if with_baseline:
        baseline = 0

        if len(cache) > 10:
            history = torch.stack(cache, dim=0)
            baseline = torch.mean(history)

            #DEBUG
            print('BASELINE ', baseline.item())
            #/DEBUG

        cache.append(torch.stack(rewards, dim=0))

        if len(cache) > 20:
            cache.pop(0)

    for step, (prob, reward) in enumerate(zip(probs, rewards)):  # Jeweils ein Schritt für alle Trajektorien im Batch
        if with_baseline:
            reward = reward - baseline

        total = total + discount**step * reward * prob

    total = torch.sum(total) / batch_size

    loss = -total
    loss.backward()
    optimizer.step()

    #DEBUG
    print('LOSS ', loss.item())
    #/DEBUG


"""
Gegeben ein Batch von Positionen und einer Policy werden Aktionen ausgewählt und im Environment ausgeführt.
param policy: Das Policy Netzwerk
param positions: Ein Batch (als Liste) von Feld-IDs. (Nummer des Feldes, wo sich der Agent befindet)
param hidden: Der hidden state des Policy-RNNs
"""
def step(policy, positions, hidden=None):
    directions = [get_valid_directions(p) for p in positions]
    batch = to_batch(directions)

    if hidden is not None:
        policies, hidden = policy(batch, hidden)

    else:
        policies, hidden = policy(batch)

    # Sample Aktionen (Indizes) aus der aktuellen Policy

    distributions = torch.distributions.Categorical(policies)
    actions = distributions.sample()
    probs = distributions.log_prob(actions)

    # Transformation der Aktionen in Strings (left, up ...) 

    actions = [direction_strings[index.item()] for index in actions]
    rewards = torch.zeros((batch_size,), device=device)
    next_positions = []

    # Ausführen der Aktionen und Feedback speichern

    for i, (action, position) in enumerate(zip(actions, positions)):
        next_position, reward = move(action, position)

        rewards[i] = reward
        next_positions.append(next_position)

    return next_positions, probs, rewards, hidden


"""
Eine Monte-Carlo Simulation, die den Wert eines Zustandes approximieren soll.
param policy: Die Policy der während der Simulation gefolgt werden soll.
param hidden: Der hidden state des Policy Netzes.
param positions: Ein Batch von Positionen (Feld_IDs), für die wir den Wert approximieren wollen.
param simulations: Anzahl der Simulationen, die wir machen. Am Ende wird über alle Simulationen gemittelt.
param steps: Anzahl der Schritte, die wir pro Simulation machen
param current_reward: Die Belohnung für den Schritt der uns in die aktuelle Position gebracht hat.
"""
def montecarlo(policy, hidden, positions, simulations, steps, current_reward):
    with torch.no_grad():
        rewards = torch.zeros((simulations, batch_size), device=device)

        for s in range(simulations):
            simulated_rewards = torch.zeros((0, batch_size), device=device)

            for i in range(steps):  # steps
                positions, _, reward, hidden = step(policy, positions, hidden)
                simulated_rewards = torch.cat((simulated_rewards, reward[None, :]), dim=0)

            rewards[s] = torch.sum(simulated_rewards, dim=0) + current_reward

        rewards = torch.mean(rewards, dim=0)

    return rewards


"""
Diese Methode geht nun einen Schritt weiter: Die Umgebung gibt uns nicht mehr nach jeder Aktion einen 
Reward, wie in der basic.py:train_with_policy_gradient() Funktion. Stattdessen müssen wir diesen über
Simulationen ermitteln.

Außerdem weiß der Agent nun nicht mehr, auf welchem Feld er sich befindet. In den Methoden zuvor haben
wir für die Zustandskodierung einen Onehot Vektor der Länge 36 für 36 Felder verwendet. Nun geben wir
dem Netz nur noch einen Onehot Vektor der Länge 4, für den gilt, dass Index i = 1, gdw. i frei und 
i = 0, wenn sich in Richtung i eine Mauer befindet.

Wir verwenden deshalb statt eines einfachen Feed-Forward Netzes ein rekurrentes Netz, mit der Idee, 
dass die Policy gegeben einen Zustand von der bisherigen Trajektorie abhängt (sonst ließen sich zwei
Felder mit identischen "Ausgängen" ja auch nicht unterscheiden).

Die Funktionen unterscheiden sich im Wesentlichen nicht: Dazugekommen ist der Aufruf der montecarlo()
Funktion, statt ein Abruf des Feld-Values.
"""

def train():
    policy = Policy().to(device)
    rollout = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learnrate)

    for epoch in range(epochs):
        rollout.set_parameters_to(policy)  # Kopie des Netzes für die MC-Simulation
        policy.train()
        rollout.eval()  # Sehr wichtig in PyTorch, wenn ihr ein Netz nutzt, dass man nicht trainieren will!

        position = [env.entry_id for _ in range(batch_size)]  # Die Startpositionen für einen Batch
        hidden = None
        probs = []
        rewards = []

        #DEBUG
        render_positions = {i: [] for i in range(36)}
        #/DEBUG
        
        for current_step in range(max_steps):
            position, prob, reward, hidden = step(policy, position, hidden)

            #missing_steps = max_steps - (current_step + 1)
            simulation = montecarlo(rollout, hidden, position, simulations, 20, reward)

            #DEBUG
            for sample in range(batch_size):
                pos = position[sample]
                val = simulation[sample].item()
                render_positions[pos].append(val)
            #/DEBUG

            rewards.append(simulation)
            probs.append(prob)

        policy_gradient(optimizer, probs, rewards)

        #DEBUG
        prettyprint([len(item) for item in render_positions.values()])
        prettyprint([numpy.mean(item) for item in render_positions.values()])
        print('SUCESS CASES ', position.count(env.exit_id), ' of ', batch_size)
        print('=========== FINISHED RUN ', epoch, ' ===========\n')
        #/DEBUG


if __name__ == '__main__':
    train()
