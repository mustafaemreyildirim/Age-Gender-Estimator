'''
usage:
    train.py  --data=<str>      [options]

options:
    --lr=<float>                learning rate [default: 0.01]
    --wd=<float>                weight decay [default: 1e-4]
    --batch_size=<int>          batch size [default: 32]
    --checkpoint=<str>          checkpoint dirname [default: model.pth]
    --resume=<str>              resume path [default: none]
    --epoch=<int>               numbers of epochs[default: 10]
    --num_works=<int>           number of workers[default: 2]
    --log=<str>                 name of summary[default: log]
    --print_freq=<int>          print frequency[default: 10]
'''
from docopt import docopt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np

from loss import AgeGenderLoss

from model import Model
from dataloader import IMDBWIKI
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw

import sys
import shutil
import time
import os


def main(args):
    image_transform =  transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    train_dataset = IMDBWIKI(root=args['--data'], transform=image_transform)
    validation_dataset = IMDBWIKI(root=args['--data'], train=False, transform=image_transform)
    display_dataset = IMDBWIKI(root=args['--data'], train=False, dp=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(args['--batch_size']), 
        shuffle=True, 
        num_workers=int(args['--num_works'])
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=int(args['--batch_size']), 
        num_workers=int(args['--num_works'])
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Model(8).to(device=device)
    loss_fn = CrossEntropyLoss().to(device=device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(args['--lr']),
        weight_decay=float(args['--wd'])
    )

    epoch, global_step = 0, 0
    resume = args['--resume']
    
    if resume and os.path.isfile(resume): 
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']

    print_freq = int(args['--print_freq'])

    print(optimizer)
    writer = SummaryWriter(args['--log'])
    for epoch in range(epoch, int(args['--epoch'])):
        global_step = train(train_dataloader, model, loss_fn, optimizer, device, writer, epoch, global_step, print_freq)
        validate(model, validation_dataloader, device, writer, epoch, global_step)

        show_image(display_dataset, model, image_transform, device, writer, global_step)
        torch.save({
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch+1,
            'global_step':global_step
            
        }, 'model_best.pth.tar')

def train(dataloader, model, loss_fn, optimizer, device, writer, epoch, global_step, print_freq):
    
    age_meter = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(dataloader),
        [age_meter],
        prefix="Epoch: [{}]".format(epoch)
    )
    
    model.train()
    
    for i, (inputs,targets) in enumerate(dataloader):
        pred = model(inputs.to(device=device))

        targets = [
            target.to(device=device, dtype=torch.float)
            for target in targets
        ]

        loss = loss_fn(pred, targets)
        

        writer.add_scalar('Loss/Loss', loss.item(), global_step)

        age_meter.update(loss.item(), inputs.size(0))
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if global_step % print_freq == 0:
            progress.display(i)

        global_step +=1
    return global_step

        

def validate(model, dataloader, device, writer, epoch, global_step):
    age_acc = AverageMeter('AgeAcc', ':6.2f')

    progress = ProgressMeter(
        len(dataloader),
        [age_acc],
        prefix="Epoch: [{}]".format(epoch)
    )
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            pred = model(inputs.to(device=device))
            gender, age = [
                target.to(device=device, dtype=torch.float)
                for target in targets
            ]

            gender_pred = pred[:, 0]
            
            n = inputs.size(0)
            gender_acc.update(
                torch.sum(~((gender > 0.5)  ^ (gender_pred > 0.5))).item() / n,
                n
            )

            age_acc.update(
                torch.mean(torch.abs(age_pred - age)),
                n
            )

            progress.display(i)

    writer.add_scalar('Acc/Gender', gender_acc.avg, global_step)
    writer.add_scalar('Acc/Age', age_acc.avg, global_step)


"""Average class for computing values"""
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



"""Progress Meter"""
class ProgressMeter(object):    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def show_image(dataset, model, transform, device, writer, global_step):
    model.eval()
    for i in range(len(dataset)):
        image, (gender, age) = dataset[i]
        pred = model(torch.unsqueeze(transform(image).to(device=device), 0))[0]
        gender_pred, age_pred = float(pred[0].item()), float(pred[1].item())

        age *= 100.0
        age_pred *= 100.0

        if gender_pred < 0.5:
            gender_pred = 0.0
        else:
            gender_pred = 1.0

        image = transforms.functional.resize(image, 64)
        draw = ImageDraw.Draw(image)
        
        draw.text(
            (5, 5),
            text=("gp: {0:d}\ng: {1:d}\nap: {2:d}\na: {3:d}".format(
                int(gender_pred),
                int(gender),
                int(age_pred),
                int(age))), 
            align ="left"
        )

        image = np.transpose(
             np.array(image),
             (2, 0, 1)
        )
        writer.add_image(
            'Image/image_{}'.format(i), 
           image, 
           global_step
        )

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
