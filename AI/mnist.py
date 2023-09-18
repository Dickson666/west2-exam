import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Running on:" + device + "......")

if not os.path.exists("./models"):
    os.makedirs("./models")

batch_size = 64
epoch = 10
learning_rate = 1e-3

training_data = datasets.MNIST(
    root = "./data",
    train = True,
    download = True,
    transform= ToTensor()
)

test_data = datasets.MNIST(
    root = "./data",
    train = False,
    download = True,
    transform = ToTensor()
)

'''
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(cols * rows):
    sample_idx = torch.randint(len(training_data), size = (1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i + 1)
    plt.title(str(label))
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''

class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    '''
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    '''
training_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

st_idx = 0
model_idx = 0
model = CNN().to(device)
if(os.path.exists("logs.txt")):
    st_idx = int(open("logs.txt", "r").read())
    model.load_state_dict(torch.load("./models/model_" + str(int(st_idx / 10) * 10) + ".pth"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

def train_loop(dataloader, model, ep):
    model.train()
    for i, (img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        res = model(img)
        loss = criterion(res, label)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{i + 1:>6d} / {len(dataloader) :>6d}] Loss: {loss :> 7f}')
            open("log.txt", "a").write(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{i + 1:>6d} / {len(dataloader.dataset) :>6d}] Loss: {loss :> 7f}\n')
def test_loop(dataloader, model):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            res = model(img)
            test_loss += criterion(res, label).item()
            correct += (res.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)
    print(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{test_loss :> 8f} \n')
    open("log.txt", "a").write(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{test_loss :> 8f} \n')

for i in range(st_idx, epoch):
    train_loop(training_dataloader, model, i)
    test_loop(test_dataloader, model)
    if (i + 1) % 10 == 0:
        if os.path.exists("./models/model_" + str(i + 1 - 10) + ".pth"):
           os.remove("./models/model_" + str(i + 1 - 10) + ".pth")
           model_idx = i + 1
        torch.save(model.state_dict(), "./models/model_" + str(i + 1) + ".pth")
        
        open("logs.txt", "w").write(str(i + 1))
