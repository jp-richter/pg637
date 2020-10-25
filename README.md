# pg637_labyrinth

## Minimal-Beispiel: Neuronales Netz

Für die Policy-Gradient Aufgabe solltet ihr unter Anderem ein Neuronales Netz verwenden.
Um euch damit vertraut zu machen, gibt es in diesem Branch ein kleines Beispiel, wie das
Training eines NNs in PyTorch aussieht.

Für das Beispiel nutzen wir den [MNIST-Datensatz](https://en.wikipedia.org/wiki/MNIST_database). Der hat sich quasi zum Hello World Programm des 
Machine Learnings entwickelt. Der Datensatz enthält Bilder von handgeschriebenen Ziffern und jedes
Bild ist entsprechend gelabelt.


![Ein Bild aus dem MNIST Datensatz mit einer handgeschriebenen 5](mnist_example.png "Ein Bild aus dem MNIST Datensatz mit einer handgeschriebenen 5")

Wir wollen für unser kleines Beispiel keine Convolutional-Layer nutzen sondern nur einfache fully-connected-Layers.
Dafür transformieren wir die Bilder aus ihrer ursprünglichen zwei-dimensionalen Form (28x28) zu
eindimensionalen Vektoren (1x784).
Dann definieren wir ein Netz das Inputvektoren dieser Größe bekommt und einen Vektor mit 10 Werten ausgibt.
Diese Werte sind die Wahrscheinlichkeiten, mit denen das Netz die entsprechenden Labels für die Eingabe vergibt.
D. h. ist der Wert an der 0. Stelle im Ausgabevektor eine 0.7 denkt das Netz, dass die Eingabe zu 70% eine 0 ist.

mnist.py enthält eine Implementierung eines kleinen Netzes, einer train-loop für das Netz und 
eine kleine Inferenz Demo des fertigen Netzes. Die spannenden Teile des Codes habe ich kommentiert, sodass hoffentlich verständlich wird, was passiert.

Wenn ihr das Skript ausführt sollte, es ein Netz trainieren und anschließend zufällig ausgewählte Bilder aus dem Testdatensatz klassifizieren.

Viel Spaß beim angucken und ausprobieren :)

### Generelle Dinge zu PyTorch

#### Tensoren

Der Dreh- und Angelpunkt von PyTorch sind Tensoren. Das sind im Prinzip Arrays mit beliebig vielen
Dimensionen. Ein 1D-Tensor ist ein Vektor, ein 2D-Tensor eine Matrix und für alles weitere gibt es 
keine speziellen Namen. Man sagt dann nur noch 3D-Tensor, 4D-Tensor usw.

*Tensoren sind sehr sehr ähnlich zu den arrays in numpy, falls jemand damit schonmal gearbeitet hat.*

Auf und mit Tensoren kann man alle möglichen mathematischen Operationen ausführen. Die Möglichkeiten findet ihr [hier](https://pytorch.org/docs/stable/torch.html). Die PyTorch Doku kann ich 
generell empfehlen. Ich finde sie sehr übersichtlich.

Dort findet ihr unter Anderem eine Zusammenfassung der [Layer](https://pytorch.org/docs/stable/nn.html#convolution-layers), [Aktivierungsfunktionen](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity), [Zielfunktionen](https://pytorch.org/docs/stable/nn.html#loss-functions) oder [Optmimierungsverfahren](https://pytorch.org/docs/stable/optim.html), die PyTorch anbietet.

Es gibt auch ein sehr gutes [Blitz-Tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html), falls jemand sich ein bisschen tiefer einlesen möchte.

#### Tipps und Tricks

In einem neuronalen Netz arbeiten wir immer in Batches. D. h. wir forwarden eigentlich nicht einen Vektor einen Tensor der Form [784] sondern einen Tensor der Form [Nx784]. Wobei N die Batch-Size ist.
Wenn wir also nur ein Bild forwarden wollen müssen wir dem entsprechenden Vektor die Form [1x784] geben.

*Um Dimensionen der Größe 1 hinzuzufügen/wegzunehmen, sind die Funktionen unsqueeze() und squeeze() sehr nütztlich:*
```
>>> import torch
>>> a = torch.Tensor([1,6,4,8,6])
>>> a
tensor([1., 6., 4., 8., 6.])
>>> a.shape
torch.Size([5])
>>> a = a.unsqueeze(0)
>>> a.shape
torch.Size([1, 5])
>>> a = a.squeeze()
>>> a.shape
torch.Size([5])
```
