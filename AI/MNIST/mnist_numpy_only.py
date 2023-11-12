import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

training_data = datasets.MNIST(
    root="./data",
    train=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="./data",
    train=False,
    transform=ToTensor()
)

batch_size = 64
learning_rate = 1e-3
epoch = 10

training_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

class Linear():
    grad_W = None
    grad_b = None
    
    def __init__(self, in_features: int, out_features: int, learning_rage :int):
        self.W = np.random.random((in_features, out_features))
        self.b = np.zeros((1, out_features))
        self.learning_rate = learning_rage
    
    def forward(self, x):
        res = x @ self.W + self.b
        self.input = x
        return res
    
    def backward(self, dy):# dy 前面的导数， 链式法则
        self.grad_W = dy @ self.input
        self.grad_b = dy
        # print(self.W.shape)
        dx = self.W @ dy
        return dx
    
    def optim(self):
        # print(self.W.shape, self.grad_W.shape)
        self.W -= learning_rate * self.grad_W.T
        self.b -= learning_rate * self.grad_b.T
        self.grad_W = 0
        self.grad_b = 0


class ReLU():
    input = None
    def __init__(self):
        pass

    def forward(self, x):
        res = x
        self.input = x
        res[res < 0] = 0
        return res

    def backward(self, dy):
        dx = np.zeros_like(dy)
        dx[self.input.T > 0] = dy[self.input.T > 0]
        return dx

def flatten(x):
    return x.reshape(x.shape[0], -1)

class NN():

    logs = {0:0}
    def __init__(self):
        self.logs.update({self.logs[0] + 1 : Linear(28 * 28, 512, learning_rate)})
        self.logs[0] += 1
        self.logs.update({self.logs[0] + 1 : ReLU()})
        self.logs[0] += 1
        self.logs.update({self.logs[0] + 1 : Linear(512, 512, learning_rate)})
        self.logs[0] += 1
        self.logs.update({self.logs[0] + 1 : ReLU()})
        self.logs[0] += 1
        self.logs.update({self.logs[0] + 1 : Linear(512, 10, learning_rate)})
        self.logs[0] += 1
    def forward(self, x):
        x = flatten(x)
        for i in range(1, self.logs[0] + 1):
            x = self.logs[i].forward(x)
        return x
    def backward(self, loss):
        pre = loss
        for i in range(0, self.logs[0]):
            pre = self.logs[self.logs[0] - i].backward(pre)

    def step(self):
        for i in range(1, self.logs[0] + 1):
            if type(self.logs[i]) == Linear:
                self.logs[i].optim()
    
model = NN()

def train_loop(dataloader, model, ep):
    for idx, (img, label) in enumerate(dataloader):
        # print(label)
        img = img.numpy()
        label = label.numpy()
        # print(label[0])
        # plt.imshow(img[0].squeeze(), cmap="gray")
        # plt.show()
        Loss = 0
        for i in range(batch_size):
            lb = np.zeros((10, 1))
            lb[label[i]] = 1
            x = model.forward(img[i])
            # print(x.shape, lb.shape)
            loss = 2 * (x.reshape((10, 1)) - lb)
            # print(loss.shape, loss)
            Loss += np.abs(loss).sum().item()
            model.backward(loss)
            model.step()
        # if(idx + 1 % 10 == 0):
        print(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{idx + 1:>6d} / {len(dataloader) :>6d}] Loss: {Loss :> 7f}')
        
def test_loop(dataloader, model):
    Loss = 0.0
    correct = 0.0
    for img, label in dataloader:
        img = np.array(img)
        label = np.array(label)

        for i in range(batch_size):
            x = model.forward(img[i])
            loss = 2 * (x - label[i])
            Loss += np.abs(loss).sum().item()
            correct += (x.argmax(1) == label[i]).type(np.float).sum().item()
    Loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)
    print(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{Loss :> 8f} \n')

for i in range(epoch):
    train_loop(training_dataloader, model, i)
    test_loop(test_dataloader, model)