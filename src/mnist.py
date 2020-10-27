import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

OUTPUT_INTERVAL = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Zuerst wollen wir unser Netz definieren.
# Ein Netz ist in PyTorch eine Klasse, die von nn.Module erbt.
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Mit nn.Sequential können wir eine Menge von Layern definieren,
        # die in der angegebenen Reihenfolge ausgeführt werden.
        # Wir definieren 3 fully-connected Layer. Hinter die letzte Layer packen wir
        # noch eine Softmax-Funktion, die dafür sorgt das alle Outputs zusammensummiert 1 ergeben.
        # Wir wollen schließlich Wahrscheinlichkeiten ausgeben.
        self.layers = nn.Sequential(
            nn.Linear(784, 128), # 1. Argument: Größe der Eingabe, 2. Argument: Größe der Ausgabe
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.LogSoftmax(dim=1) # Wir nehmen Log von Softmax, weil es rechnerisch schöner ist und besser zum Loss passt.
        )

    # In der forward-Funktion wird standardmäßig definiert, wie Eingaben für das Netz verarbeitet
    # werden und wie das Netz Ausgaben produziert.
    # Das ist in unserem Fall aber nicht sehr aufwendig...
    def forward(self, x):
       # Wir jagen einfach den Input durch alle Layer. 
       return self.layers(x)


# Als nächstes definieren wir uns eine Funktion mit der wir die Daten für unser Problem holen
def load_data():
    # Lädt den MNIST-Datensatz runter und speichert in im Ordner "data".
    # Außerdem kriegen wir ein Dataset-Objekt zurück, mit dem wir theoretisch schon über die 
    # einzelnen Bilder und entsprechenden Label iterieren könnten.
    trainset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    # Ein Dataloader ist ein praktischer Aufsatz auf ein Dataset, um in Mini-Batches über unserern
    # Datensatz zu iterieren. Ein Iterator von diesem Objekt gibt uns zum Beispiel bei jedem Schritt
    # 32 Bilder und die dazugehörigen Labels.
    # Das hier ist der Trainingsdatensatz. Damit sich also die Trainingsepochen voneinander unterscheiden, 
    # setzen wir das shuffle-Flag und bei jeder Epoche gibt uns der DataLoader verschiedene
    # Batches in verschiedener Reihenfolge.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True)

    # Jetzt laden wir auch noch den Testdatensatz runter.
    testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
    # Den müssen wir allerdings nicht shufflen.
    testloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=False)
    return trainloader, testloader


# Kommen wir zur eigentlichen Trainingsroutine
def train_loop(net, trainset, epoch):
    
    # Wir wählen unser Optimierungsverfahren (in unserem Fall Stochastic Gradient Descent
    # mit einer Lernrate von 0.01).
    # Das optimiser-Objekt führt später die updates der Gewichte für uns aus. 
    # Deswegen übergeben wir ihm auch Referenzen auf die Gewichte (Parameter) des Netzes.
    optimiser = torch.optim.SGD(net.parameters(), lr=0.01)
    # Anschließend wählen wir die Loss-Function in unserem Fall Negative-log-likelihood.
    criterion = torch.nn.NLLLoss()

    # Damit wir wissen was passiert loggen wir den Loss im Laufe des Trainings
    running_loss = 0.0
    
    # Wir iterieren Batchweise einmal über den ganzen Datensatz.
    for i, data_batch in enumerate(trainloader):
        # Gradienten der (Netz)parameter vor jedem neuen Update nullen
        optimiser.zero_grad()
        inputs, labels = data_batch[0].to(device), data_batch[1].to(device)
        batch_size = len(inputs)
        # Mnist hat Bilder der Dimension 28x28. Wir arbeiten ausschließlich mit fully-connected Layern.
        # Deswegen machen wir aus einem Batch von zwei-dimensionalen Tensoren einen Batch
        # von ein-dimensionalen Tensoren.
        inputs = inputs.reshape(batch_size, 784)
        # Hier benutzen wir unser Netz
        # Dadurch das wir von nn.Module erben überschreiben wir auch den ()-Operator mit der 
        # forward-Funktion. net(inputs) ist also das gleiche wie net.forward(inputs)
        outputs = net(inputs)
        # Wir berechnen den Wert der Zielfunktion für diesen Batch.
        loss = criterion(outputs, labels)
        # Jetzt kommen wir zum Kernstück von PyTorch:
        # PyTorch baut dynamisch während es z. B. den Output eines Netzes berechnet den
        # Computational Graph. Das heißt es weiß genau, wie z. B. der Loss berechnet wurde.
        # Mit der nächsten Zeile lassen wir PyTorch entlang des dynamisch gebauten 
        # Computational Graph (CG) die Gradienten berechnen.
        loss.backward()
        # Jetzt hat PyTorch für jedes Gewicht des Netzes (alle Gewichte sind Teil des CG)
        # den Gradienten berechnet und er optimiser kann den nächsten Update-Schritt ausfürhen.
        optimiser.step()
        
        # Hier machen wir die Ausgabe darüber wie sich der loss entwickelt.
        running_loss += loss.item()
        if i % OUTPUT_INTERVAL == OUTPUT_INTERVAL - 1:
            print("[{}, {batch:5d} loss: {loss:.3f}]".format(epoch + 1, batch=i + 1, loss=running_loss / OUTPUT_INTERVAL))
            running_loss = 0.0


def imshow(img):
    npimg = img.reshape(28, 28).cpu().numpy()
    plt.imshow(npimg, cmap='binary')
    plt.show()

def predict_random_sample(net, testloader):
    inputs, _ = iter(testloader).next()
    inputs = inputs.to(device)
    batch_size = len(inputs)
    img = inputs[random.randrange(batch_size)].reshape(1, 784)
    print("Labels:        0     1     2     3     4     5     6     7     8     9")
    pred_str = "Probabilities: "
    preds = torch.exp(net(img)).squeeze()
    for i in range(len(preds)):
        pred_str += "{:.3f} ".format(preds[i].item())
    print(pred_str)
    imshow(img)

def compute_accuracy(dataloader, net):
    net.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = len(inputs)
            inputs = inputs.reshape(batch_size, 784)
            outputs = net(inputs)
            predictions = outputs.argmax(dim=1)
            for result in torch.eq(predictions, labels):
                if result:
                    correct += 1

    print("Accuracy: {acc:.3f}".format(acc=correct/len(dataloader.dataset)))


if __name__ == "__main__":
    trainloader, testloader = load_data()
    net = Net().to(device)
    print("Accuracy with untrained net:")
    compute_accuracy(testloader, net)
    for i in range(3):
        train_loop(net, trainloader, i)
    print("Accuracy after training:")
    compute_accuracy(testloader, net)

    print("Now let's label some handdrawn digits with our trained net:")
    print()
    user_in = ""
    while user_in != "q":
        predict_random_sample(net, testloader)
        user_in = input("Enter for next one, q for exit: ")
