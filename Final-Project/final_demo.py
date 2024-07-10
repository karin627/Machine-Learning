# -*- coding: utf-8 -*-
"""Final_demo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tvaihtMmdwILYtCY7zu-WtUkoCyN4_Rq
"""

from google.colab import drive
drive.mount("/content/drive")

!unzip /content/drive/MyDrive/dataset.zip

!pip install thop
!pip install torchsummary

# Commented out IPython magic to ensure Python compatibility.
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset
from glob import glob
from natsort import natsorted
import os
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
from google.colab.patches import cv2_imshow
import random

from thop import profile
from thop import clever_format
from torchsummary import summary

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(42)

!pwd
train_adults_dir = "dataset/dataset/train/adults"
train_adults_image = natsorted(glob(os.path.join(train_adults_dir,'*.jpg')))
print(len(train_adults_image))

train_children_dir = "dataset/dataset/train/children"
train_children_image = natsorted(glob(os.path.join(train_children_dir,'*.jpg')))
print(len(train_children_image))

train_image = train_adults_image + train_children_image
print(len(train_image))

test_adults_dir = "dataset/dataset/test/adults"
test_adults_image = natsorted(glob(os.path.join(test_adults_dir,'*.jpg')))
print(len(test_adults_image))

test_children_dir = "dataset/dataset/test/children"
test_children_image = natsorted(glob(os.path.join(test_children_dir,'*.jpg')))
print(len(test_children_image))

test_image = test_adults_image + test_children_image
print(len(test_image))

transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224,224]),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(p=1),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform3 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224,224]),
    transforms.RandomRotation(30),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# transform4 = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize([224,224]),
#     transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])
transform5 = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

])

class MyDataset(Dataset):
  def __init__(self,data_names,transform_type):
    self.image_names = np.array(data_names)
    self.transform = transform_type
  def __len__(self):
    return len(self.image_names)
  def __getitem__(self,idx):
    input_name = self.image_names[idx]
    # print(input_name)
    # cv2_imshow(cv2.imread(input_name))
    # image = cv2.imread(input_name).astype(np.float32) /255.
    # image = torch.from_numpy(image.transpose(2,0,1))
    image = self.transform(cv2.imread(input_name))
    # print(image.shape)
    if input_name.find("adults") != -1:
      target = 0
    else:
      target = 1
    return {
        'input_name':input_name,
        'input_image':image,
        'target':target
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 20
# valid_size = 0.1
aug_num = 3
train_dataset1 = MyDataset(train_image,transform1)
train_dataset2 = MyDataset(train_image,transform2)
train_dataset3 = MyDataset(train_image,transform3)
combined_train_dataset = ConcatDataset([train_dataset1,train_dataset2,train_dataset3])
test_dataset = MyDataset(test_image,transform1)
# train_dataset1.__getitem__(idx = 1)

# validation
# num_train = len(combined_train_dataset)
# indices = list(range(num_train))
# np.random.shuffle(indices)
# split = int(np.floor(valid_size*num_train))
# train_idx,valid_idx = indices[split:],indices[:split]

# train_sampler = SubsetRandomSampler(train_idx)
# valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = DataLoader(combined_train_dataset,batch_size=batch_size,shuffle=True)
# trainloader = DataLoader(combined_train_dataset,batch_size=batch_size,sampler=train_sampler)
# validloader = DataLoader(combined_train_dataset,batch_size=batch_size,sampler=valid_sampler)
testloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

MODEL_NEURONS = 16
class ConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_depth),
            nn.Conv2d(in_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_depth),
            nn.Conv2d(out_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ShallowUNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.conv_down1 = ConvBlock(in_channel,MODEL_NEURONS)
        self.conv_down2 = ConvBlock(MODEL_NEURONS, MODEL_NEURONS * 2)
        self.conv_down3 = ConvBlock(MODEL_NEURONS * 2, MODEL_NEURONS * 4)
        self.conv_bottleneck = ConvBlock(MODEL_NEURONS * 4, MODEL_NEURONS * 8)
        self.conv1 = ConvBlock(128,64)
        self.conv2 = ConvBlock(64,16)

        self.linear2 = nn.Linear(16*14*14,64)
        self.linear3 = nn.Linear(64,2)

        self.se1 = SE_Block(32)
        self.se2 = SE_Block(64)
        self.se3 = SE_Block(128)

        self.maxpool = nn.MaxPool2d(2)

        self.relu = nn.ReLU()
        self.swish = Swish()
        self.dropout = nn.Dropout(p=0.2)



    def forward(self, x): # [-1, 3, 224, 224]
        x = self.conv_down1(x) # [-1, 16, 224, 224]
        x = self.conv_down2(self.maxpool(x)) # [-1, 40, 112, 112]
        x = self.conv_down3(self.maxpool(x)) # [-1, 64, 56, 56]
        x = self.dropout(x)
        x = self.conv_bottleneck(self.maxpool(x)) # [-1, 128, 28, 28]


        x = self.conv1(self.maxpool(x)) # [-1, 64, 14, 14]
        x = self.conv2(x)    # [-1, 16, 14, 14]

        x = x.view(-1,16*14*14) # [-1, 2, 196]
        x = self.linear2(x) #[-1, 21, 2]
        x = self.relu(x)
        x = self.linear3(x)
        return x

model = ShallowUNet(3,2).to(device)
print(model)

# pseudo image
image = torch.rand(1, 3, 224, 224).to(device)

out = model(image)

# torchsummary report
summary(model, input_size=(3, 224, 224))
print(f'From input shape: {image.shape} to output shape: {out.shape}')

# thop report
macs, parm = profile(model, inputs=(image,))
print(f'FLOPS: {macs * 2 / 1e9} G, Params: {parm / 1e6} M.')
flops = macs * 2 / 1e9
params = parm / 1e6

n_epochs = 12
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

traindata_num = aug_num*560
# valid_loss_min = np.Inf
history = []
for epoch in range(n_epochs):
  correct_sum = 0
  train_loss = 0.0
  # valid_loss = 0.0
  lrs = []
  result = {'train_loss':[],'lrs':[],'train_accuracy':[],'valid_loss':[]}
  model.train()
  for batch_idx,data in enumerate(tqdm(trainloader)):
    input = data['input_image'].to(device)
    target = data['target'].to(device)
    optimizer.zero_grad()
    output = model(input)
    _,pred = torch.max(output,1)
    correct = np.squeeze(pred.eq(target))
    loss = criterion(output,target)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()*data['input_image'].size(0)
    correct_sum += correct.sum()

  # validation
  # model.eval()
  # for batch_idx,data in enumerate(tqdm(validloader)):
  #   input = data['input_image']
  #   target = data['target']
  #   output = model(input)
  #   loss = criterion(output,target)
  #   valid_loss += loss.item()*data['input_image'].size(0)

  print(correct_sum)
  train_accuracy = correct_sum/traindata_num;
  train_loss = train_loss/len(trainloader.dataset)
  # valid_loss = valid_loss/len(validloader.dataset)
  result['lrs'] = lrs
  result['train_loss'] = train_loss
  result['train_accuracy'] = train_accuracy
  # result['valid_loss'] = valid_loss
  history.append(result)
  scheduler.step()
  lrs.append(optimizer.param_groups[0]['lr'])
  print('Epoch {:2d}: Learning Rate = {:.6f} Training Loss = {:.6f} Training Accuracy = {:.6f}'.format(
      epoch+1,
      lrs[-1],
      train_loss,
      train_accuracy
    ))
  if epoch == 11:
    torch.save(model.state_dict(),"weight.pt")
  # save model if validation loss has decreased
  # if valid_loss <= valid_loss_min:
  #     print("Validation loss decreased({:.6f}-->{:.6f}). Saving model ..".format(
  #     valid_loss_min,
  #     valid_loss
  #     ))
  #     torch.save(model.state_dict(),"model.pt")
  #     valid_loss_min = valid_loss

test_loss = 0.0
correct_test_sum = 0
model.load_state_dict(torch.load('weight.pt'))
model.eval()
for data in testloader:
  input = data['input_image'].to(device)
  target = data['target'].to(device)
  output = model(input)
  loss = criterion(output,target)
  test_loss += loss.item()*input.size(0)
  _,pred = torch.max(output,1)
  correct = np.squeeze(pred.eq(target))
  correct_test_sum += correct.sum()

test_loss /= len(testloader.dataset)
test_accuracy = correct_test_sum/120
print('Test Loss:{:.6f} Test Accuracy:{:.6f}'.format(test_loss,test_accuracy))