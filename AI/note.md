## MINST

### 配置环境

#### 安装CUDA

cmd 输入 `nividia-smi` 查看驱动信息，

官网下载对应版本 https://developer.nvidia.com/cuda-toolkit-archive。

#### 安装pytorch

https://pytorch.org/get-started/previous-versions/ 安装相互兼容的pytorch版本。


### 头文件

```python
import torch
import torch.nn as nn//用于搭建神经网络
import torch.optim as optim//导入优化器
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt//可视化
```

### 超参数

手动设定的一些重要参数，一般有批次数量(`batch_size`)、训练周期数(`epoch`)、学习效率(`learning_rate`)。

```python
batch_size = 64
epoch = 10
learning_rate = 1e-3
```

### 导入数据

使用datasets导入训练数据，一些相关参数：
- `root` 储存位置
- `train` 是否训练
- `download` 当指定位置没有文件时，是否进行下载
- `transform` 对数据的预处理方式

```python
training_data = datasets.MNIST(
    root = "./data", //储存位置
    train = True, //是否用于训练
    download = True,//如果没有数据是否自动下载
    transform= ToTensor()//预处理方式
)

test_data = datasets.MNIST(
    root = "./data",
    train = False,
    download = True,
    transform = ToTensor()
)
```

然后使用 `Dataloader` 进行数据预加载，相关参数：
- `dataset` 导入的数据集
- `batch_size` 单批次的数量（同时进行的数据个数）
- `shuffle` 是否打乱顺序

```python

training_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

```

### 创建神经网络模型

创建一个以 `nn.Module` 为父类的类，并在其中创建一个简单的神经网络
```python
class NN(nn.Module):
    
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
    
    def forward(self, x)://在神经网络上运行的方式
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

#### `nn.Flatten()`

扁平化操作，即把多个维度展平为一个维度。

例如，一个形状为 `(batch_size, channel, height, width)` 的张量，经过 `nn.Flatten()` 操作后，会变成一个形状为 `(batch_size, channel * height * width)` 的张量。 `batch_size` 这一维不会被展平。

#### `nn.Linear()`

全连接层（稠密层），用于实现线性变化。
通常变化为 $y = Wx + b$，其中 $y$ 为输出， $x$ 是输入， $W$ 是权重矩阵。 $b$ 是偏置矩阵。其中 $W$ 和 $b$ 是可优化的参数。

`nn.Linear` 的参数包含 `in_features` 和 `out_features`，即输入特征维度和输出特征维度。例如 `nn.Linear(114, 514)` 将接受 114 维的输入， 产生 514 维的输出。 

#### `nn.ReLU()`

激活函数层， 引入非线性性质。

这里的 `nn.ReLU` 是修正线性单元，`nn.ReLU(x) = max(0, x)`。

### Device

模型再GPU上运行速度会比在CPU上快，所有有条件应该将其搬到GPU上以提高效率。注意，要保证两个相关的数据不可以储存在两个不同的设备上。

```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = NN().to(device)//在合适的设备上创建模型
```

### 训练

计算损失时使用 `nn.CrossEntropyLoss`。`nn.CrossEntropyLoss` 是pytorch自带的一个损失计算函数，常用于分类问题。

计算公式：

$$CrossEntropyLoss(x, y) = -\sum_{i=1}y_i \times \log(x_i)$$

其中 $x$ 是神经网络的预测概率， $y$ 为真实标签。

同时还要使用 `optim.Adam`，是pytorch提供的一种优化器。
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
```
然后就是具体的学习过程和测试过程：

```python
def train_loop(dataloader, model, ep):
    model.train()
    for i, (img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)//保证设备统一
        optimizer.zero_grad()//优化器清空梯度
        res = model(img)//在模型上运行
        loss = criterion(res, label)//求loss
        loss.backward()//求梯度
        optimizer.step()//优化参数
        if (i + 1) % 100 == 0:
            print(f'Epoch [{ep + 1:>3d} / {epoch:>3d}] Step [{i + 1:>6d} / {len(dataloader) :>6d}] Loss: {loss :> 7f}')
def test_loop(dataloader, model):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            res = model(img)
            //求正确率和loss
            test_loss += criterion(res, label).item()
            correct += (res.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)
    print(f'Test Error: \n Accuracy : {(correct * 100):> 0.1f} % ,Avg loss:{test_loss :> 8f} \n')
```

最后在主程序调用：
```python
for i in range(st_idx, epoch):
    train_loop(training_dataloader, model, i)
    test_loop(test_dataloader, model)
```

#### 储存和读取

存储：
```python
torch.save(model.state_dict(), "./model.pth")
```

读取：
```python
model.load_state_dict(torch.load("./models.pth"))
```