"""
code adapted from: https://github.com/kaidic/LDAM-DRW
"""
import argparse
import os
import random
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from collections import Counter

import resnet_cifar as models
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import LDAMLoss, FocalLoss, ASLSingleLabel

############################################################

t00 = time.time()
t0=time.time()



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print('model names ',model_names)

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')

parser.add_argument('--dataset', 
                    default='cifar10', 
                    help='dataset setting')

parser.add_argument('-a', '--arch', metavar='ARCH', 
                    default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')

parser.add_argument('--loss_type', 
                    default="CE", 
                    #default = 'LDAM',
                    #default = 'Focal',
                    #default = 'ASL',
                    type=str, help='loss type')

parser.add_argument('--imb_type', 
                    default="exp", 
                    #default = 'step',
                    type=str, help='imbalance type')

parser.add_argument('--imb_factor', 
                    default=0.01, 
                    type=float, help='imbalance factor')

parser.add_argument('--train_rule', 
                    default='None',
                    #default='Reweight',
                    #default='Resample',
                    #default='DRW',
                    type=str, 
                    help='data sampling strategy for train loader')

parser.add_argument('--rand_number', 
                    default=0, #1
                    #default=10, #2
                    #default =100, #3
                    type=int, 
                    help='fix random number for data sampling')

parser.add_argument('--exp_str', default='0', type=str, 
                    help='number to indicate which experiment it is')

parser.add_argument('-j', '--workers', default=0, type=int, 
                    metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', 
                    default = 200, 
                    type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', 
                    default=128, 
                    type=int,
                    metavar='N',
                    help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float,
                    metavar='LR', help='initial learning rate',
                    dest='lr')

parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4,
                    type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', 
                    default=0,
                    type=int,
                    help='seed for initializing training. ')

parser.add_argument('--data_path',
        default='.../data/',
        type=str,
        help='data path.')

parser.add_argument('--save_file_path',
        default='.../models/',
        type=str,
        help='data path.')


parser.add_argument('--gpu', 
                    default = 0,
                    type=int,
                    help='GPU id to use.')

parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
best_acc1 = 0

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.set_printoptions(precision=4, sci_mode=False, threshold=50000)
np.set_printoptions(precision=4, suppress=True,threshold=50000)


best_acc = 0  # best test accuracy

def main():
    global t00
    args = parser.parse_args()
    
    for arg in vars(args):
        print (arg, getattr(args, arg))
    print() 
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type,
            args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    
    
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    print('n gpus ',ngpus_per_node)
    main_worker(args.gpu, ngpus_per_node, args)
    print()
    print('total time (min): %.3f\n' % ((time.time()-t00)/60))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed(0) 
    np.random.seed(0)
    random.seed(0)



def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    
    global t0
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    
    use_norm = True if args.loss_type == 'LDAM' else False
    
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    
    count = count_parameters(model)
    print('num params ',count)
    print()
    torch.cuda.manual_seed(0)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        
        train_dataset = IMBALANCECIFAR10(root=args.data_path, 
                imb_type=args.imb_type, imb_factor=args.imb_factor,
                rand_number=args.rand_number, train=True, download=True,
                transform=transform_train)
        
        print(train_dataset.data.shape, type(train_dataset.data[0]), 
              type(train_dataset.data), train_dataset.data.dtype)
        
        print(train_dataset.targets[0].dtype,
              type(train_dataset.targets))
        
        print(Counter(train_dataset.targets), len(train_dataset.targets))
        
        val_dataset = datasets.CIFAR10(root=args.data_path,
                train=False,
                download=True, transform=transform_val)
        
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root=args.data_path,
                imb_type=args.imb_type,
                imb_factor=args.imb_factor, 
                rand_number=args.rand_number,
                train=True, download=True, 
                transform=transform_train)
        val_dataset = datasets.CIFAR100(root=args.data_path,
                train=False,
                download=True, 
                transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    
    cls_num_list = train_dataset.get_cls_num_list()
    
    args.cls_num_list = cls_num_list
    
    train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, 
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        if args.train_rule == 'None':
            train_sampler = None  
            per_cls_weights = None 
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            print('rewt')
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            if epoch == 0:
                print('eff num ',effective_num)
            
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            
            if epoch == 0:
                print('per cls wts ',per_cls_weights)
            
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            
            if epoch == 0:
                print('per cls wts ',per_cls_weights)
            
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')
        
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
           
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30,
                                 weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            
            criterion = FocalLoss(weight=per_cls_weights, gamma=2).cuda(args.gpu)
            
        
        #no clip in single
        elif args.loss_type == 'ASL':
            criterion=ASLSingleLabel()
        
        else:
            warnings.warn('Loss type is not listed')
            return
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training,
              tf_writer)
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        
        if epoch % 10 == 0:
            t1 = (time.time() - t0)/60
            print('%2d epoch time (min): %.3f\n' % (epoch,t1))
            t0 = time.time()
        

def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    #global best_acc
    train_loss = 0
    correct = 0
    total = 0
    #count = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    count = 0
    train_on_gpu = torch.cuda.is_available() 
    classes = ('0', '1', '2', '3', '4','5', '6', '7', '8', '9')
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        output = model(input)
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        tar_np = target.detach().cpu().numpy()
        
        tar_len = len(tar_np)
        
        total += target.size(0)
        
        _, pred = torch.max(output, 1) 
        
        pred_np = pred.detach().cpu().numpy()
        
        if count == 0:
            y_true = np.copy(tar_np)
            y_pred = np.copy(pred_np)
        else:
            y_true = np.concatenate((y_true,tar_np),axis=None)
            y_pred = np.concatenate((y_pred,pred_np),axis=None)
        count+=1
        
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        
        if tar_len > 1:
            for n in range(tar_len):
                label = target.data[n]
                class_correct[label] += correct[n].item()
                class_total[label] += 1 
                
        else:
            for n in range(tar_len):
                label = target.data[n]
                class_correct[label] += correct.item()#[n]
                class_total[label] += 1 
        
        if i % 25 == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,
                    lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()
        
    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    
    if epoch % 10 == 0:
        for i in range(10):
            if class_total[i] > 0:
                print('Train Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Train Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        print('\nTrain Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    
        target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4',
            'class 5', 'class 6', 'class 7', 'class 8','class 9'] 

        gm = geometric_mean_score(y_true, y_pred, average='macro') 
        pm = precision_score(y_true, y_pred, average='macro')
        fm = f1_score(y_true, y_pred, average='macro',zero_division=1)
        acsa = accuracy_score(y_true, y_pred) #acsa
        bacc = balanced_accuracy_score(y_true, y_pred)
        
        print('ACSA ',acsa)
        print('bacc ',bacc)
        print('GM ',gm)
        print('PM ',pm)
        print('FM ',fm)
        
def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None,
             flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    global best_acc
    train_loss = 0
    correct = 0
    total = 0
    #count = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    count = 0
    train_on_gpu = torch.cuda.is_available() 
    
    classes = ('0', '1', '2', '3', '4','5', '6', '7', '8', '9')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            tar_np = target.detach().cpu().numpy()
            
            tar_len = len(tar_np)
            
            total += target.size(0)
        
            _, pred = torch.max(output, 1) 
        
            pred_np = pred.detach().cpu().numpy()
            
            if count == 0:
                y_true = np.copy(tar_np)
                y_pred = np.copy(pred_np)
            else:
                y_true = np.concatenate((y_true,tar_np),axis=None)
                y_pred = np.concatenate((y_pred,pred_np),axis=None)
            count+=1
        
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

            for i in range(tar_len):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1 
        
            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)


        if epoch % 1 == 0:
            for i in range(10):
            
                if class_total[i] > 0:
                    print('Validation Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
                else:
                    print('Validation Accuracy of %5s: N/A (no training examples)' % (classes[i]))

            print('\nValidation Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
    
            target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4',
            'class 5', 'class 6', 'class 7', 'class 8','class 9'] 


            gm = geometric_mean_score(y_true, y_pred, average='macro') 
            pm = precision_score(y_true, y_pred, average='macro')
            fm = f1_score(y_true, y_pred, average='macro',zero_division=1)
            acsa = accuracy_score(y_true, y_pred) #acsa
            bacc = balanced_accuracy_score(y_true, y_pred)
        
            print('ACSA ',acsa)
            print('bacc ',bacc)
            print('GM ',gm)
            print('PM ',pm)
            print('FM ',fm)
        
            if fm > best_acc:
                print('Saving..')
                
                sfile = args.save_file + args.loss_type + \
                    '_C10_' + str(epoch) +  '_' + args.train_rule + '_best.pth'
                
                torch.save(model.state_dict(), sfile)
                best_acc = fm

    return top1.avg

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()