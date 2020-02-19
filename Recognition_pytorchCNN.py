import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.optim import Adam


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 5, 5),
      nn.ReLU(),
      nn.MaxPool2d(3, stride = 3),
      nn.BatchNorm2d(5)
      )

    self.layer2 = nn.Sequential(
      nn.Conv2d(5, 15, 3),
      nn.ReLU(),
      nn.MaxPool2d(2, stride = 2),
      nn.BatchNorm2d(15)
      )

    self.layer3 = nn.Sequential(
      nn.Conv2d(15, 25, 3),
      nn.ReLU(),
      nn.MaxPool2d(2, stride = 2),
      nn.BatchNorm2d(25)
      )

    self.layer4 = nn.Sequential(
      nn.Conv2d(25, 35, 3),
      nn.ReLU(),
      nn.MaxPool2d(2, stride = 2),
      nn.BatchNorm2d(35)
      )

    self.fc = nn.Sequential(
      nn.Linear(3*4*35, 200),
      nn.ReLU(),
      nn.Linear(200, 10)
      )
    
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    
    x = x.view(-1, 3*4*35)
    x = self.fc(x)
    x = torch.softmax(x, dim=-1)
    
    return x

def accuracy(argmax, label):
  cnt = 0
  for i in range(len(label)):
    if(argmax[i] == label[i]):
      cnt += 1
  return cnt / len(label)

img_size = 129*144
learning_rate =0.001
num_epoch = 30
batch_size = 40
total_time = 45

mag_train_data = np.load('recordings\\mag_train_data.npy')
ang_train_data = np.load('recordings\\ang_train_data.npy')
mag_test_data = np.load('recordings\\mag_test_data.npy')    
ang_test_data = np.load('recordings\\ang_test_data.npy')
label_train = np.load('recordings\\label_train.npy')
label_test = np.load('recordings\\label_test.npy')

model = CNN().cuda()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([np.prod(p.size()) for p in model_parameters])
print("number of parameters : {}".format(num_params))

optimizer = Adam(model.parameters(), lr = learning_rate)

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epoch):
    total_loss = 0.0
    
    for trial in range(total_time):
        optimizer.zero_grad()
        
        input = mag_train_data[trial*40 : trial*40 + 40]
        input = torch.from_numpy(input).type(torch.FloatTensor)
        input = input.cuda()

        label = label_train[trial*40 : trial*40 + 40]
        label = torch.from_numpy(label).type(torch.LongTensor)
        _, label = label.max(dim = -1)
        res = model(input)
        loss = criterion(res, label.cuda())
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print("Epoch: "+ str(epoch+1)+ "  Total loss : {}".format(total_loss))

torch.save(model.state_dict(), "CNN.pt")

test_time = 5

model_test = CNN().cuda()
model_test.load_state_dict(torch.load("CNN.pt"))
model_test.eval()

for trial in range(test_time):
    
    input = mag_test_data[trial*40 : trial*40 + 40]
    input = torch.from_numpy(input).type(torch.FloatTensor)
    input = input.cuda()

    label = label_test[0:40]
    label = torch.from_numpy(label).type(torch.LongTensor)

    _, label = label.max(dim = -1)

    res = model_test(input)
    _, argmax = res.max(dim=-1)
    test_acc = accuracy(argmax,label)

    print("Acc : {}".format(test_acc))

