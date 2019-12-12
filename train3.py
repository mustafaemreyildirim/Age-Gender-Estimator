'''
usage:
    train.py  --data=<str>      [options]

options:
    --lr=<float>                learning rate [default: 0.01]
    --decay=<float>             weight decay [default: 0.0]
    --batch_size=<int>          batch size [default: 32]
    --checkpoint=<str>          checkpoint dirname [default: model.pth]
    --resume=<str>              resume path [default: none]
    --number_epochs=<int>       numbers of epochs[default: 10]
    --num_works=<int>           number of workers[default: 4]
    --start_epoch=<int>         start epoches[default: 0]

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
import shutil
import time
import os


args = docopt(__doc__)

dir = str(args['--data'])
batchSize = int(args['--batch_size'])
epochs = int(args['--number_epochs'])
learningRate = float(args['--lr'])
decay = float(args['--decay'])
resume = str(args['--resume'])
checkpoint = str(args['--checkpoint'])
num_workers = int(args['--num_works'])
start_epoch = int(args['--start_epoch'])

best_acc1 = 0

def main():
    
    global best_acc1

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ]),
    'val': transforms.Compose([
        transforms.Resize((64,64)),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    trainset = IMDBWIKI(root=dir, transform=data_transforms['train'])
    valset = IMDBWIKI(root=dir, train=False, transform=data_transforms['val'])

    net = Model(8)
    loss_fn = AgeGenderLoss(0.42,0.58)
    optimizer = optim.Adam(net.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08, weight_decay=decay, amsgrad=False)

    
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
        
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    
    for epoch in range(start_epoch, epochs):
        # train for one epoch
        train_epoch(net, loss_fn, optimizer, epoch, trainset)

    # evaluate on validation set
        validate( net, loss_fn, valset)

    # remember best acc@1 and save checkpoint
        #is_best = acc1 > best_acc1
        #best_acc1 = max(acc1, best_acc1)


        save_checkpoint({
            'epoch': epoch + 1,
            'checkpoint': net.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        })


def train_epoch(net,loss_fn,optimizer,epoch,dataset_train):
    
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batchSize, shuffle= True, num_workers=num_workers)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    
    net.train()
    end = time.time()
    for i, (inputs,targets) in enumerate(dataloader):
        data_time.update(time.time() - end)
       
        outputs = net(inputs)
        age_loss,gender_loss = loss_fn(outputs,targets)
        
        
        total_loss = age_loss+gender_loss
        
       
        losses.update(total_loss.item(), inputs.size(0))
        
        optimizer.zero_grad()
        total_loss.backward()

        optimizer.step()
        batch_time.update(time.time() - end)
        progress.display(i)

def validate( model, criterion, valset):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(valset),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    total = 0
    total_age = 0
    age_correct = 0
    gender_correct = 0


    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(valset):


            # compute output
            output = model(inputs)
            loss = criterion(output, target)

            # measure accuracy and record loss
           # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            age_correct_i, gender_correct_i = accuracy(output, target)

           # top1.update(acc1[0], inputs.size(0))
           # top5.update(acc5[0], inputs.size(0))
            total_age += age_correct_i
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            total += len(inputs)

            if i % args.print_freq == 0:
                progress.display(i)

            print('age_prec: %.3f (%d/%d) | gender_prec: %.3f  [%d/%d]' ,   
               age_correct/total,  \
               age_correct/total,  \
               i+1, len(valset))

    

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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



def accuracy(output, targets):
    '''Measure batch accuracy.'''
    AGE_TOLERANCE = 0.05
    
    gender_output, age_output, = output[:, 0, None], output[:, 1]
    gender_target, age_targets = targets
    age_correct = ((1-(gender_output-age_targets).abs() < AGE_TOLERANCE)/gender_target*100).long().sum().data[0]

    gender_correct = (1-(gender_output - gender_target)/gender_target*100).long().sum().data[0]
    return age_correct, gender_correct


if __name__ == '__main__':
    main()
