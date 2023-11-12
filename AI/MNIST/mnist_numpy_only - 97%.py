import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化图像数据
])

training_data = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform
)
test_data = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform
)


batch_size = 64
learning_rate = 1e-3
epoch = 100

training_dataloader = DataLoader(training_data, batch_size, True)
test_dataloader = DataLoader(test_data, batch_size, True)

def softmax(x):
    # mx = None
    mx = np.max(x)
    # print(mx)
    # print(mx)
    sum = np.log(np.sum(np.exp(x - mx)))
    # print(sum)
    res = x - mx - sum
    # res = np.exp(res)
    return res


class Linear():
    grad_W = None
    grad_b = None
    
    def __init__(self, in_features: int, out_features: int, learning_rage :int):
        self.W = np.random.normal(scale= 0.1, size=(in_features, out_features))
        # self.W = softmax(self.W)
        self.b = np.zeros((out_features))
        self.learning_rate = learning_rage
    
    def forward(self, x):
        
        res = x @ self.W
        for i in range(len(res)):
            res[i] += self.b
        self.input = x
        # open("loss.txt", "a").write("IIIIIIII\n" + str(x) + "\nSSSSSSSSSSSSSSSSSSSSSSSS\n" + str(res) + "\n AAAAA \n")
        return res
    
    def backward(self, dy):# dy 前面的导数， 链式法则
        self.grad_W = dy @ self.input
        self.grad_b = dy
        # open("loss.txt", "a").write(str(dy) + "\nQWQWQWQW\n"+str(self.W)+"\n\n\n"+str(self.grad_W)+"\n----------------------------------\n")
        # print(self.W.shape)
        dx = self.W @ dy
        return dx
    
    def optim(self):
        # print(self.W.shape, self.grad_W.shape)
        self.W -= self.learning_rate * self.grad_W.T
        for i in range(len(self.grad_b.T)):
           self.b -= self.learning_rate * self.grad_b.T[i]

        self.learning_rate *= 0.99994

        print(f"learning_rate={self.learning_rate:>.5f}", end=",")
        # self.W = softmax(self.W)
        # self.b = softmax(self.b)
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
        # open("loss.txt", "a").write(str(res) + "\n BBBBBBBBBB \n")
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

def CrossEntropyLoss(x, y, z):
    # print(x.shape, y.shape)
    res = 0
    for j in range(z):
        for i in range(len(x)):
            res -= y[i][j].item() * x[i][j].item()
            # print(x[i][j], y[i][j], res)
            # time.sleep(1)
    return res

def train_loop(dataloader, model, ep):
    for idx, (img, label) in enumerate(dataloader):
        # print(label)
        img = img.numpy()
        label = label.numpy()
        if len(label) != batch_size:
            continue
        # print(label[0])
        # plt.imshow(img[0].squeeze(), cmap="gray")
        # plt.show()
        Loss = 0
        lb = np.zeros((10, len(label)))
        for i in range(len(label)):
            lb[label[i]][i] = 1
        x = model.forward(img)
        # print(x)
        # print(x)
        for i in range(batch_size):
            x[i] = softmax(x[i])
        # x = softmax(x)
        # print(x)
        # time.sleep(1)
        # loss = 2 * ((x.T) - lb)
        # loss = -lb / np.maximum(np.exp(x.T), 1e-10)
        loss = np.exp(x.T) - lb
        # open("loss.txt", "a").write(str(loss.T) + "\n********************\n")
        Loss = CrossEntropyLoss(x.T, lb, len(label))
        model.backward(loss)
        model.step()
        
        '''
        for i in range(len(label)):
            lb = np.zeros((10, 1))
            lb[label[i]] = 1
            x = model.forward(img[i])
            # print(x.shape, lb.shape)
            loss = 2 * (x.reshape((10, 1)) - lb)
            # print(loss.shape, loss)
            Loss += CrossEntropyLoss(x.T, lb)
            model.backward(loss)
            model.step()
        '''
        Loss /= len(label)
        # print(type(Loss), Loss)
        # if(idx + 1 % 10 == 0):
        print(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{idx + 1:>6d} / {len(dataloader) :>6d}] Loss: {Loss :> 7f}')
        # open("log.txt", "a").write(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{idx + 1:>6d} / {len(dataloader) :>6d}] Loss: {Loss :> 7f}\n')
        
def test_loop(dataloader, model):
    Loss = 0.0
    correct = 0.0
    for img, label in dataloader:
        img = np.array(img)
        label = np.array(label)

        if len(label) != batch_size:
            continue
        lb = np.zeros((10, len(label)))
        for i in range(len(label)):
            lb[label[i]][i] = 1
        x = model.forward(img)
        for i in range(batch_size):
            x[i] = softmax(x[i])
        Loss += CrossEntropyLoss(x.T, lb, len(label))
        correct += (x.argmax(1) == label).sum().item()
    # Loss /= len(label)
    Loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)
    print(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{Loss :> 8f} \n')
    open("log.txt", "a").write(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{Loss :> 8f} \n')

for i in range(epoch):
    train_loop(training_dataloader, model, i)
    test_loop(test_dataloader, model)