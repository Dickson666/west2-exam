import torch
from utils.dataloader import yoloDataset, yoloCollate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from loss import Computeloss
import torch.optim as optim
from models.yolo import YOLO
from trainer import train, test

batch_size = 16
lr = 3e-4
epoch = 100
anchors = [[10, 13, 16, 30, 33, 23],
           [30, 61, 62, 45, 59, 119],
           [116, 90, 156, 198, 373, 326]]
anchors[0] = [i / 8 for i in anchors[0]]
anchors[1] = [i / 16 for i in anchors[1]]
anchors[2] = [i / 32 for i in anchors[2]]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([640, 640])
])

train_dataset = yoloDataset("./data/VOCdevkit/VOC2012", True, transform = trans)
test_dataset = yoloDataset("./data/VOCdevkit/VOC2012", False, transform = trans)

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = yoloCollate)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, collate_fn = yoloCollate)

model = YOLO().to(device)

crit = Computeloss(anchors, 20, device = device)
optims = optim.SGD(model.parameters, lr)

for i in range(epoch):
    train(model, i, train_dataloader, crit, optims, device, epoch)
    test(model, test_dataloader, crit, device)