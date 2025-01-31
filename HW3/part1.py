# -*- coding: utf-8 -*-
"""hw3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yK5ai7F4uqCMppsrNQjR2Zip9wqltPv4
"""

from google.colab import drive
drive.mount("/content/drive")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
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
torch.backends.cudnn.enchmark = False
torch.use_deterministic_algorithms(True)

transform = transforms.Compose([
    transforms.ToTensor()
])

# problem = 1

training_batch_size = 16
testing_batch_size = 64

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=training_batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=testing_batch_size, shuffle=False)

# problem 2
# data_size = 40000
# subset_indices = torch.arange(data_size)
# subtrain = torch.utils.data.Subset(trainset, subset_indices)
# trainloader = torch.utils.data.DataLoader(subtrain, batch_size=training_batch_size, shuffle=True)

# problem 3
layer = 1

# problem 1, 2
nodes = 100
class DNN(nn.Module):
  def __init__(self):
    super(DNN, self).__init__()
    self.fc1 = nn.Linear(28*28, nodes)
    self.fc_hidden = nn.Linear(nodes, nodes)
    self.fc2 = nn.Linear(nodes, 10)

  def forward(self, x):
    # flatten image input
    x = x.view(-1, 28*28)
    x = F.relu(self.fc1(x))
    for i in range(layer-1):
      x = F.relu(self.fc_hidden(x))
    x = self.fc2(x)
    return x

model = DNN()
print(model)

# specify loss function
criterion = nn.CrossEntropyLoss()

# number of epochs to train the model
n_epochs = 10

# specify optimizer
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# results = []
# print('Learning Rate: {:.4f}\n'.format(lr))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for epoch in range(n_epochs):
  # train_loss = 0.0

  model.train()

  for data, target in trainloader:
    optimizer.zero_grad()
    # forward
    output = model(data)
    # calculate loss
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    # train_loss += loss.item()*data.size(0)

    if epoch==n_epochs-1:
      _, pred = torch.max(output, 1)
      correct = np.squeeze(pred.eq(target.data.view_as(pred)))
      for i in range(target.data.size(0)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

#   train_loss = train_loss/len(trainloader.dataset)
#   results.append(train_loss)
#   print('Epoch {:2d}: Training Loss: {:.6f}'.format(epoch+1, train_loss))
# print('\n')

# calcualte training accuracy
# for i in range(10):
#   if class_total[i] > 0:
#     print('Training Accuracy of Class %2s: %2.2f%% (%2d/%2d)' % (
#         str(i),
#         100*class_correct[i]/class_total[i],
#         np.sum(class_correct[i]),
#         np.sum(class_total[i])
#     ))
#   else:
#     print('Training Accuracy of Class %2s: N/A')

# f = open('/content/drive/MyDrive/school/ml_hw3_result/prob'+str(problem)+'_'+str(training_batch_size)+'_lr'+str(lr)+'_train.txt', 'w')
# f.write('Training Accuracy:'+str(100*np.sum(class_correct)/np.sum(class_total)))

print('\nTraining Accuracy: %2.2f%% (%2d/%2d)' % (
    100*np.sum(class_correct)/np.sum(class_total),
    np.sum(class_correct),
    np.sum(class_total)
))

# def plot_loss(results):
#   iterations = np.arange(len(results))
#   plt.plot(iterations, results, '-bx')
#   plt.xlabel('epoch')
#   plt.ylabel('loss')
#   plt.legend(['Training'])
#   plt.title('Loss vs. number of epochs')
#   plt.show()

# plot_loss(results)

# testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()

for data, target in testloader:
  output = model(data)
  loss = criterion(output, target)
  _, pred = torch.max(output, 1)
  correct = np.squeeze(pred.eq(target.data.view_as(pred)))
  # print(target.data.size())
  for i in range(target.data.size(0)):
    label = target.data[i]
    class_correct[label] += correct[i].item()
    class_total[label] += 1

# calcualte accuracy
# for i in range(10):
#   if class_total[i] > 0:
#     print('Testing Accuracy of Class %2s: %2.2f%% (%2d/%2d)' % (
#         str(i),
#         100*class_correct[i]/class_total[i],
#         np.sum(class_correct[i]),
#         np.sum(class_total[i])
#     ))
#   else:
#     print('Testing Accuracy of Class %2s: N/A')

# f.write('\nTesting Accuracy:'+str(100*np.sum(class_correct)/np.sum(class_total)))
# f.close()

print('\nTesting Accuracy: %2.2f%% (%2d/%2d)' % (
    100*np.sum(class_correct)/np.sum(class_total),
    np.sum(class_correct),
    np.sum(class_total)
))