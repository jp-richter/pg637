import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_INTERVAL = 500


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.layer_one = nn.Linear(784, 128)
        self.layer_two = nn.Linear(128, 32)
        self.layer_three = nn.Linear(32, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
       x = self.relu(self.layer_one(x))
       x = self.relu(self.layer_two(x))
       x = self.layer_three(x)
       return self.softmax(x)

def compute_accuracy(dataloader, net):
    net.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            batch_size = len(inputs)
            inputs = inputs.reshape(batch_size, 784)
            outputs = net(inputs)
            predictions = outputs.argmax(dim=1)
            for result in torch.eq(predictions, labels):
                if result:
                    correct += 1

    print("Accuracy: {acc:.3f}".format(acc=correct/len(dataloader.dataset)))



def load_data():
    trainset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True)
    testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=False)
    return trainloader, testloader

def train_loop(net, trainset, epoch):

    optimiser = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()
    running_loss = 0.0

    for i, data_batch in enumerate(trainloader):
        optimiser.zero_grad()
        inputs, labels = data_batch
        batch_size = len(inputs)
        inputs = inputs.reshape(batch_size, 784)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        
        running_loss += loss.item()
        if i % OUTPUT_INTERVAL == OUTPUT_INTERVAL - 1:
            print("[{}, {batch:5d} loss: {loss:.3f}]".format(epoch + 1, batch=i + 1, loss=running_loss / OUTPUT_INTERVAL))
            running_loss = 0.0


def imshow(img):
    npimg = img.reshape(28, 28).numpy()
    plt.imshow(npimg, cmap='binary')
    plt.show()

def predict_random_sample(net, testloader):
    inputs, _ = iter(testloader).next()
    img = inputs[10].reshape(1, 784)
    print("Labels:        0     1     2     3     4     5     6     7     8     9")
    pred_str = "Probabilities: "
    preds = torch.exp(net(img)).squeeze()
    for i in range(len(preds)):
        pred_str += "{:.3f} ".format(preds[i].item())
    print(pred_str)
    imshow(img)



if __name__ == "__main__":
    trainloader, testloader = load_data()
    net = Net()
    compute_accuracy(testloader, net)
    for i in range(1):
        train_loop(net, trainloader, i)
    compute_accuracy(testloader, net)
    predict_random_sample(net, testloader)
