import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from PIL import Image
import random
import math
from torchsummary import summary

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if not os.path.exists("./models"):
    os.makedirs("./models")

batch_size = 2
epoch = 10
patch_size = 16
learning_rate = 1e-3

transfer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class MyDataset(Dataset):

    def __init__(self, dir, transform) -> None:
        # print(dir)
        super().__init__()
        self.dir = str(dir),
        # print(os.listdir(dir))
        filenames = [f for f in os.listdir(dir)]
        self.lab = []
        self.filenames = []
        for i in filenames:
            dirr = os.path.join(dir, i)
            for j in os.listdir(dirr):
                self.lab.append(i)
                self.filenames.append(j)
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)

    # def getlab(self, dir):
    #     return os.path.dirname(dir)
    
    def __getitem__(self, index) -> torch.tensor:
        # print(type(self.dir), type(self.lab[index]), type(self.filenames[index]))
        dir = self.dir[0] + "/" + self.lab[index] + "/" + self.filenames[index]
        # print(dir)
        data = Image.open(dir).convert("RGB")
        if(self.transform):
            data = self.transform(data)
        return data, self.lab[index]
        
dataset = MyDataset('./data/caltech101/101_ObjectCategories', transfer)

# print(len(dataset))
# exit(0)

num = [i for i in range(len(dataset))]

random.shuffle(num)

train_len = int(len(dataset) * 0.8)

train_dataset = []
test_dataset = []

for i in range(train_len):
    train_dataset.append(dataset[num[i]])
for i in range(train_len, len(num)):
    test_dataset.append(dataset[num[i]])

print(len(train_dataset), len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)


# exit(0)


class Pre_work(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.Tensor(batch_size, 1, patch_size ** 2 * 3))
        self.emd = nn.Parameter(torch.Tensor(batch_size, 196, patch_size ** 2 * 3))
        self.linear = nn.Linear(768, 768)
    
    def forward(self, x):
        # print(x.shape)
        x = x.view(batch_size, 3, 224, -1, 16)
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        # print(x.shape)
        x = x.view(batch_size, 3, x.shape[2], x.shape[3], -1, 16)
        # print(x.shape)
        x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
        x = x.view(batch_size, x.shape[1], x.shape[2], -1)
        x = x.view(batch_size, -1, x.shape[3])
        # print(x.shape, "/n", self.emd.shape)
        # print(x.device, self.emd.device)
        return self.linear(torch.cat([x + self.emd, self.cls], dim = 1))

def trans_pos(x, n):
    x = x.view(batch_size, 197, n, -1)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(-1, x.shape[2], x.shape[3])
    return x
def trans_neg(x):
    x = x.view(batch_size, -1, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(batch_size, x.shape[1], -1)
    return x

class dotproductattention(nn.Module):
    def __init__(self, dropout = 0.5) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
    
    def forward(self, q, v, k):
        d = q.shape[-1]
        res = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        res = nn.functional.softmax(res)
        return torch.bmm(self.drop(res), v)

class multihead(nn.Module):
    def __init__(self, q_size, v_size, k_size, head_num, hide_num) -> None:
        super().__init__()
        self.attention = dotproductattention()
        self.W_q = nn.Linear(q_size, hide_num)
        self.W_v = nn.Linear(v_size, hide_num)
        self.W_k = nn.Linear(k_size, hide_num)
        self.head_num = head_num
        
    def forward(self, q, v, k):
        Q = trans_pos(self.W_q(q), self.head_num)
        K = trans_pos(self.W_k(k), self.head_num)
        V = trans_pos(self.W_v(v), self.head_num)
        res = self.attention(Q, V, K)
        return trans_neg(res)

class vit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre = Pre_work()
        self.multihead = multihead(768, 768, 768, 12, 768)
        self.l1 = nn.LayerNorm((197, 768))
        self.l2 = nn.LayerNorm((197, 768))
        self.linear = nn.Linear(768, 101)
        self.mlp = nn.Sequential(
            nn.Linear(768, 3072),
            nn.Linear(3072, 768)
        )
    
    def forward(self, x):
        # print(x.shape)
        x = self.pre(x)
        print(x.shape)
        y = self.l1(x)
        y = self.multihead(y, y, y)
        y = y + x
        z = self.l2(y)
        z = self.mlp(z)
        z = z + y
        return self.linear(torch.squeeze(z[:, 0, :]))

model = vit().to(device)
summary(model, (3, 224, 224), batch_size=batch_size, device=device)

optims = optim.Adam(model.parameters(), lr = learning_rate)
crit = nn.CrossEntropyLoss()

def train(dataloader, model, ep):
    model.train()
    for i, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        print(image.shape, label.shape)
        optims.zero_grad()
        res = model(image)
        loss = crit(res, label)
        loss.backward()
        optims.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{(i + 1) :>4d} / {len(dataloader):>4d}] Loss: {loss :>7f}')

def test(dataloader, model):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            res = model(img)
            loss = crit(res, label)
            test_loss += loss.item()
            correct += (res.argmax(1) == label).type(torch.float).sum().item()
    correct /= len(dataloader.dataset)
    test_loss /= len(dataloader)
    print(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{test_loss :> 8f} \n')

if __name__ == "__main__":
    for i in range(epoch):
        train(train_dataloader, model, i)
        test(test_dataloader, model)