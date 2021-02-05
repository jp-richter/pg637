import torch
import torch.nn
import numpy
import environment as env

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 16
learnrate = 0.2
discount = 0.9
epochs = 100
length = 50


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
            torch.nn.LogSoftmax(dim=1) 
        )

    def forward(self, x):
       return self.layers(x)


def floyd():
    distances = numpy.full((36,36), 1000)

    for i in range(36):
        distances[i, i] = 0

        dirs = env.get_valid_directions(i)
        neighbors = [env.move(d, i)[0] for d in dirs]

        for j in neighbors:
            distances[i, j] = 1

    for i in range(36):
        for j in range(36):
            for k in range(36):
                if distances[j, k] > distances[j, i] + distances[i, k]:
                    distances[j, k] = distances[j, i] + distances[i, k]

    return distances


distances = floyd()
values = [0 for _ in range(36)]

for i in range(36):
    dexit = distances[i, env.exit_id]
    dtrap = distances[i, env.trap_id]

    if i == env.exit_id:
        values[i] = 1
        continue

    if i == env.trap_id:
        values[i] = -1
        continue

    value = (dtrap - dexit) / (dtrap + dexit)
    values[i] = value


def onehots(positions):
    t_positions = torch.zeros((batch_size, 36), device=device)

    for i, p in enumerate(positions):
        t_positions[i][p] = 1

    return t_positions


def best_move(position):
    best_action = None
    best_value = -10

    for direction in env.get_valid_directions(position):
        target, _ = env.move(direction, position)
                    
        if values[target] > best_value:
            best_action = direction
            best_value = values[target]

    return best_action


"""
In dieser Variante setzen wir perfekte Information beim Agenten voraus. Er kennt jeweils die Abstände
zum Ausgang und zur Falle. Aus den beiden Distanzen, ermitteln wir einen Value für jedes Feld. Im 
Idealfall entscheidet sich der Agent immer für das Feld mit der größten Value Differenz zum jetzigen Feld.

Wir formulieren das Problem als Regression, bei dem der Agent gegeben einen eindeutigen Zustand predicten
soll, welches Zielfeld das beste im Sinne der Aufgabe ist. Da wir wissen, welches Feld das beste ist, 
haben wir für eine Policy auch ein Label. Angenommen, das beste Feld ist rechts vom Agenten, dann wäre
bei einer Policy von etwa [0.2, 0.2, 0.4, 0.2] das Label der Index 1. Für dieses Setting können wir die
Kreuzentropie minimieren.
"""

def train_with_crossentropy():
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=learnrate)
    criterion = torch.nn.NLLLoss()

    for epoch in range(epochs):
        positions = [env.entry_id for _ in range(batch_size)]

        #DEBUG
        render_positions = [0 for i in range(36)]
        #/DEBUG

        for step in range(length):
            input = onehots(positions)
            policy = net(input)

            distributions = torch.distributions.Categorical(torch.exp(policy))
            actions = distributions.sample()
            actions = [env.direction_strings[index.item()] for index in actions]

            labels = torch.zeros((batch_size,), device=device, dtype=torch.long)
            pos_tmp = []

            for i, (action, position) in enumerate(zip(actions, positions)):
                target, _ = env.move(action, position)
                pos_tmp.append(target)

                #DEBUG
                render_positions[target] += 1
                #/DEBUG
                
                best_action_string = best_move(position)
                best_action_index = env.direction_indices[best_action_string]

                labels[i] = best_action_index

            positions = pos_tmp

            optimizer.zero_grad()
            loss = criterion(policy, labels)
            loss.backward()
            optimizer.step()

            #DEBUG
            env.prettyprint(render_positions)
            #/DEBUG


"""
In dieser Variante setzen wir perfekte Information beim Agenten voraus. Er kennt jeweils die Abstände
zum Ausgang und zur Falle. Aus den beiden Distanzen, ermitteln wir einen Value für jedes Feld. Im 
Idealfall entscheidet sich der Agent immer für das Feld mit der größten Value Differenz zum jetzigen Feld.

Jetzt betrachten wir die Values der Felder als Teil der Umgebung, der für uns nicht tractable ist. Der
Agent bekommt zwar noch Belohnungen, kann aber nicht im Voraus sagen, welche Belohnungen er bekommen wird.
In unserem Beispiel ist natürlich klar, dass sich die Belohnung nie ändern wird. Dieser Ansatz lässt
sich aber auf alle Fälle abstrahieren, in dem der Agent nach jedem Schritt von der Umgebung einen Reward 
bekommt. Wir sparen uns in diesem Ansatz die aufwändige Simulation und die Approximation eines Zustandsvalues.
Diese Methode stellt also eine Art Zwischenschritt zwischen der Cross Entropy Methode und dem REINFORCE
mit Monte Carlo Simulation in reinforce.py dar.
"""

def train_with_policy_gradient():
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=learnrate)

    for epoch in range(epochs):
        positions = [env.entry_id for _ in range(batch_size)]
        probs = torch.empty((length, batch_size), device=device)
        rewards = torch.empty((length, batch_size), device=device)

        #DEBUG
        render_positions = [0 for i in range(36)]
        #/DEBUG

        for step in range(length):
            input = onehots(positions)
            policy = net(input)

            distributions = torch.distributions.Categorical(policy)
            actions = distributions.sample()
            probs_tmp = distributions.log_prob(actions)

            actions = [env.direction_strings[index.item()] for index in actions]
            rew_tmp = torch.zeros((batch_size), device=device)
            pos_tmp = []

            for i, (action, position) in enumerate(zip(actions, positions)):
                target, _ = env.move(action, position)
                rew_tmp[i] = values[target] - values[position]
                pos_tmp.append(target)

                #DEBUG
                render_positions[target] += 1
                #/DEBUG

            probs[step] = probs_tmp
            rewards[step] = rew_tmp
            positions = pos_tmp

        total = torch.zeros((batch_size,), device=device)
        optimizer.zero_grad()

        for step, (p, r) in enumerate(zip(probs, rewards)):
            total = total + discount**step * r * p

        total = torch.sum(total) / batch_size

        loss = -total
        loss.backward()
        optimizer.step()

        #DEBUG
        env.prettyprint(render_positions)
        #/DEBUG
