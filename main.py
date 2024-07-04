import torch
import torchvision
import data_handler
from torchvision import datasets
from Resnet import ResNet
from torch import nn

learning_rate = 1e-3
batch_size = 128

train_dataloader, test_dataloader = data_handler.self_dataset(batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

block1 = nn.Sequential(
    nn.Conv2d(1, 64 ,kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

def resnet_block(input_channels, num_channels, num_residuals, first_block=True):
    blk = []
    for i in range(num_residuals):
        if i == 0 and first_block == False:
            blk.append(ResNet(input_channels, num_channels,use_1x1conv=True))
        else:
            blk.append(ResNet(num_channels, num_channels))
    return blk

block2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
block3 = nn.Sequential(*resnet_block(64, 128, 2))
block4 = nn.Sequential(*resnet_block(128, 256, 2))
block5 = nn.Sequential(*resnet_block(256, 512, 2))

model = nn.Sequential(
    block2, block2, block3, block4, block5,
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(), nn.Linear(512, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer =torch.optim.AdamW(model.parameters(),lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    epochs = 10
    print("Current device is", device)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == '__main__':
    main()