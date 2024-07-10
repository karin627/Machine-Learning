import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import os

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

train_batch_size = 16
test_batch_size = 64

train_file = pd.read_csv('HW2_training.csv')
train_features = torch.tensor(train_file[['Offensive', 'Defensive']].values).to(torch.float32)
train_targets_raw = train_file['Team'].values
for i in range(train_targets_raw.shape[0]):
    if(train_targets_raw[i]==0):
        train_targets_raw[i]=3
train_targets = torch.tensor(train_targets_raw).to(torch.long)

test_file = pd.read_csv('HW2_testing.csv')
test_features = torch.tensor(test_file[['Offensive', 'Defensive']].values).to(torch.float32)
test_targets_raw = test_file['Team'].values
for i in range(test_targets_raw.shape[0]):
    if(test_targets_raw[i]==0):
        test_targets_raw[i]=3
test_targets = torch.tensor(test_targets_raw).to(torch.long)

class CustomDataSet(Dataset):
    def __init__(self, features, targets=None):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feature = self.features[index]
        target = self.targets[index]
        return feature, target

train_data = CustomDataSet(train_features, train_targets)
trainloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_data = CustomDataSet(test_features, test_targets)
testloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

class DNN(nn.Module):
  def __init__(self):
    super(DNN, self).__init__()
    self.fc1 = nn.Linear(2, 4)
    self.fc_hidden0 = nn.Linear(4, 8)
    self.fc_hidden1 = nn.Linear(8, 16)
    self.fc_hidden2 = nn.Linear(16, 32)
    self.fc_hidden3 = nn.Linear(32, 64)
    self.fc_hidden4 = nn.Linear(64, 64)
    self.fc2 = nn.Linear(64, 4)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc_hidden0(x))
    x = F.relu(self.fc_hidden1(x))
    x = F.relu(self.fc_hidden2(x))
    x = F.relu(self.fc_hidden3(x))
    x = F.relu(self.fc_hidden4(x))
    x = self.fc2(x)
    return x

model = DNN()

# specify loss function
criterion = nn.CrossEntropyLoss()

# number of epochs to train the model
n_epochs = 10

# specify optimizer
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def decision_boundary():
  p = np.zeros((0,2))
  tp = np.zeros((0,1))
  for i in range(101):
    for j in range(101):
      p = np.vstack((p, [i,j]))
      tp = np.vstack((tp, [0]))
      
  prediction = []
      
  model.eval()
  points = torch.tensor(p).to(torch.float32)
  tpoints = torch.tensor(tp).to(torch.float32)
  boundary_data = CustomDataSet(points, tpoints)
  loader = DataLoader(boundary_data, batch_size=750, shuffle=False)
  for data, target in loader:
    output = model(data)
    _, pred = torch.max(output, 1)
    for i in range(target.size(0)):
      prediction.append(pred[i])
  
  for cls in range(4):
    points = np.squeeze(np.take(p, np.where(np.array(prediction) == cls), axis=0))
    if cls == 0:
        color = "red"
    elif cls == 1:
        color = "blue"
    elif cls == 2:
        color = "green"
    elif cls == 3:
        color = "yellow"
    plt.scatter(points[:, 0], points[:, 1], color = color)

  plt.xlabel('x1 : offensive')
  plt.ylabel('x2 : defensive')
  plt.legend()
  plt.title('PartII  DNN Model')
  plt.show()
print('Learning Rate: {:.4f}\n'.format(lr))

confusion = np.zeros([4, 4])

for epoch in range(n_epochs):
  model.train()

  for data, target in trainloader:
    optimizer.zero_grad()
    # forward
    output = model(data)
    # calculate loss
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if epoch==n_epochs-1:
      _, pred = torch.max(output, 1)
      for i in range(target.size(0)):
        confusion[target[i]][pred[i].item()] += 1

# calcualte training accuracy
confusion = np.delete(confusion, 0, 1)
confusion = np.delete(confusion, 0, 0)
print('training confusion')
print(confusion)

print('Training Accuracy: %2.2f%%' % (
    100.*np.sum(np.diagonal(confusion))/np.sum(confusion)
))

# testing
test_loss = 0.0
confusion = np.zeros([4, 4])

model.eval()

for data, target in testloader:
  # print(data)
  output = model(data)
  loss = criterion(output, target)
  test_loss += loss.item()*data.size(0)
  _, pred = torch.max(output, 1)
  for i in range(target.size(0)):
    confusion[target[i]][pred[i].item()] += 1
    

# calcualte accuracy
print('\ntesting confusion')

confusion = np.delete(confusion, 0, 1)
confusion = np.delete(confusion, 0, 0)
print(confusion)

print('Testing Accuracy: %2.2f%%' % (
    100.*np.sum(np.diagonal(confusion))/np.sum(confusion)
))

decision_boundary()