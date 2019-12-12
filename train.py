'''
usage:
    train.py  --data=<str>      [options]

options:
    --lr=<float>                learning rate [default: 0.01]
    --decay=<float>             weight decay [default: 0.0]
    --batch_size=<int>          batch size [default: 32]
    --checkpoint=<str>          checkpoint dirname [default: model.pth]
    --resume=<bool>             resume [default: False]
    --number_epochs=<int>       numbers of epochs[default: 10]
    --num_works=<int>           number of workers[default: 4]

'''
from docopt import docopt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loss import AgeGenderLoss
from network import Model
from dataloader import IMDBWIKI
from torchvision import transforms

import sys

args = docopt(__doc__)

dir = str(args['--data'])
batchSize = int(args['--batch_size'])
numEpochs = int(args['--number_epochs'])
learningRate = float(args['--lr'])
decay = float(args['--decay'])
resume = bool(args['--resume'])
checkpoint = str(args['--checkpoint'])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = IMDBWIKI(root=dir, transform=transform_train)





net = Model(8)
loss_fn = AgeGenderLoss(0.42,0.58)
optimizer = optim.Adam(net.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08, weight_decay=decay, amsgrad=False)

def train_epoch(model,loss_fn,optimizer,dataset_train):
    net.train()
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batchSize, shuffle= True, num_workers=0)
    for i, (inputs,targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = net(inputs)
        age_loss,gender_loss = loss_fn(outputs,targets)
        print('age loss: {},gender loss: {}'.format(age_loss,gender_loss))
        total_loss = age_loss+gender_loss
        total_loss.backward()
        optimizer.step()
        print('it: {},total loss: {}'.format(i*batchSize,total_loss.item()))

for a in range(numEpochs):
    train_epoch(net,loss_fn,optimizer,trainset)
    model.save_state_dict(checkpoint)
