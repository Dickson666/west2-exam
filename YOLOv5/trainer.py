import torch

def train(model, epoch, dataloader, crit, optim, device, ep):
    model.train()
    for i, (img, target) in enumerate(dataloader):
        img = img.to(device)
        target = target.to(device)
        optim.zero_grad()
        res = model(img)
        loss, each_loss = crit(res, target)
        loss.backward()
        optim.step()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1:>3d} / {ep:>3d}] Step [{(i + 1) :>4d} / {len(dataloader):>4d}] Loss: {loss :>7f} lr: {optim.param_groups[0]["lr"]}')

def test(model, dataloader, crit, device):
    model.eval()
    Loss = 0
    for img, target in enumerate(dataloader):
        img = img.to(device)
        target = target.to(device)
        res = model(img)
        loss, _ = crit(res, target)
        Loss += loss
    Loss /= len(dataloader)
    print(f'Test: \n Avg loss:{Loss :> 8f}')