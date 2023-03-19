#code reference: Fisher Yu and Eric Smith, https://github.com/fyu/drn

import sys
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import shutil 

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from helpers import FocalLoss, adjust_learning_rate, compute_metrics, save_checkpoint, AverageMeter, count2
from augment import aug_fake, remove_fake
import drn_binary as models

class run_training():
    """ Main training code """
    def prepare_data(self, args):
        """ prepare data according to options (add fake data or not) """
        # if add fake patches: (aug = 1)
        if args.aug == 1:
            print('**')
            print('********** Preparing dataset for training augmented model... ')
            print('Using pseudo labeling threshold of ', args.aug_threshold)
            aug_fake(args)
        
        elif args.aug != 1:
            # make sure that the dataset does not contain trans_patches
            print('**')
            print('********** Preparing dataset for training baseline mode... ')
            remove_fake(args)

    def check_data(self, args):
        """ Double check the data """
        count2(args.path_data)
        
        
    def main_train(self, args):
        
        # create model
        model = models.__dict__[args.arch](args.pretrained)
        model = torch.nn.DataParallel(model).cuda()
    
        best_auc = 0
        best_f1 = 0
        train_loss = []
        val_loss = []
        sens_list = []
        spec_list = []
        f1_list = []
        acc_list = []
        auc_list = []
    
        cudnn.benchmark = True
    
        # Data loading code
        traindir = os.path.join(args.path_data, 'train')
        valdir = os.path.join(args.path_data, 'val')
        #normalize using imagenet's mean and std
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                #transforms.RandomResizedCrop(args.scale_size),
                transforms.Resize(args.scale_size),
                transforms.CenterCrop(args.crop_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(args.scale_size),
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        
        criterion = FocalLoss().cuda()
    
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        
        timestr = time.strftime("%Y%m%d-%H%M")
        if args.aug == 1:
            args.savepath = os.path.join(args.savepath,'classify_{}_aug_{}_{}_{}_lr_{}_best_{}/'.format(timestr, 
                                            args.aug_type, args.loss_name, args.opt_name, args.lr, args.best_name))
        elif args.aug == 0:
            args.savepath = os.path.join(args.savepath,'classify_{}_{}_{}_lr_{}_best_{}/'.format(timestr, 
                                            args.loss_name, args.opt_name, args.lr, args.best_name))
        if not os.path.exists(args.savepath):
            os.makedirs(args.savepath)
    
        for epoch in range(args.epochs):
            adjust_learning_rate(args, optimizer, epoch)
            t_loss = train(args, train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            v_acc, v_loss, v_sens, v_spec, v_f1, v_auc = validate(args, val_loader, model, criterion)

            if args.best_name == 'auc':
                #remember best roc auc and save checkpoint as model best
                is_best = v_auc > best_auc
                best_auc = max(v_auc, best_auc)
            elif args.best_name == 'f1':
                #use f1 as save best criterion
                is_best = v_f1 > best_f1
                best_f1 = max(v_f1, best_f1)
    
            checkpoint_path = args.savepath + 'checkpoint_latest.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'rocauc': v_auc,
                'f1': v_f1
            }, is_best, args, filename=checkpoint_path)
            
            train_loss.append(t_loss)
            val_loss.append(v_loss)
            
            sens_list.append(v_sens)
            spec_list.append(v_spec)
            f1_list.append(v_f1)
            acc_list.append(v_acc)
            auc_list.append(v_auc)
    
            if (epoch + 1) % args.check_freq == 0:
                history_path = args.savepath + 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
                shutil.copyfile(checkpoint_path, history_path)
                
                print('*******************PLOT HERE*******************')
                plt.figure(figsize  = (8,5))
                plt.plot(range(len(train_loss)), train_loss, label = 'train')
                plt.plot(range(len(val_loss)), val_loss, label = 'val')
                plt.xlabel('epoch', fontsize = 14)
                plt.ylabel('losses', fontsize = 14)
                plt.legend()
                plt.title('training and val losses', fontsize = 14)
                plt.savefig(args.savepath +'losses_ep' + str(epoch) + '.png')
                plt.show()
    
                plt.figure(figsize  = (8,5))
                plt.plot(range(len(sens_list)), sens_list, label = 'val_sensitivity')
                plt.plot(range(len(spec_list)), spec_list, label = 'val_specificity')
                plt.plot(range(len(f1_list)), f1_list, label = 'val_f1')
                plt.plot(range(len(acc_list)), acc_list, label = 'val_acc')
                plt.plot(range(len(auc_list)), auc_list, label = 'val_auc')
                plt.xlabel('epoch', fontsize = 14)
                plt.ylabel('metrics', fontsize = 14)
                plt.legend()
                plt.title('training and val metrics', fontsize = 14)
                plt.savefig(args.savepath +'metrics_ep' + str(epoch) + '.png')
                plt.show()
                    

def train(args, train_loader, model, criterion, optimizer, epoch):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    sens = AverageMeter()
    spec = AverageMeter()
    f1s = AverageMeter()
    accs = AverageMeter()

    y_label = []
    y_pred_prob = []

    # switch to train mode
    model.train()

    sta = time.time()
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        # predicted prob
        pred_prob = F.softmax(output)

        y_label.extend(list(target.cpu().numpy()))
        y_pred_prob.extend(list(pred_prob.cpu().detach().numpy()[:,1]))

        #print(output.data, target)
        loss = criterion(output, target_var)

        # compute metrics and record loss
        sensitivity, specificity, f1, acc = compute_metrics(output.data, target)
        losses.update(loss.data.item(), input.size(0))
        sens.update(sensitivity, input.size(0))
        spec.update(specificity, input.size(0))
        f1s.update(f1, input.size(0))
        accs.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - sta)
        sta = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Sensitivity {sens.val:.3f} ({sens.avg:.3f})\t'
                  'Specificity {spec.val:.3f} ({spec.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   loss=losses, sens = sens, spec = spec, f1 = f1s, acc = accs))
            
    auc = roc_auc_score(y_label, y_pred_prob)

    print('**** Epoch: [{0}]\t'
          'ROC AUC {auc:.3f}\t'.format(epoch, auc = auc))

    return losses.avg

def validate(args, val_loader, model, criterion):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    sens = AverageMeter()
    spec = AverageMeter()
    f1s = AverageMeter()
    accs = AverageMeter()

    y_label = []
    y_pred_prob = []

    model.eval()

    sta = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        pred_prob = F.softmax(output)

        y_label.extend(list(target.cpu().numpy()))
        y_pred_prob.extend(list(pred_prob.cpu().detach().numpy()[:,1]))

        loss = criterion(output, target_var)
        sensitivity, specificity, f1, acc = compute_metrics(output.data, target)
        losses.update(loss.data.item(), input.size(0))
        sens.update(sensitivity, input.size(0))
        spec.update(specificity, input.size(0))
        f1s.update(f1, input.size(0))
        accs.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - sta)
        sta = time.time()

        if i % args.print_freq == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Sensitivity {sens.val:.3f} ({sens.avg:.3f})\t'
                  'Specificity {spec.val:.3f} ({spec.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, 
                   sens = sens, spec = spec, f1 = f1s, acc = accs))
            
    #compute roc auc
    auc = roc_auc_score(y_label, y_pred_prob)
    
    print(' * Sensitivity {sens.avg:.3f} Specificity {spec.avg:.3f} F1 {f1s.avg:.3f} Acc {accs.avg:.3f} ROC AUC {auc:.3f}'
          .format(sens = sens, spec = spec, f1s = f1s, accs = accs, auc = auc))

    return accs.avg, losses.avg, sens.avg, spec.avg, f1s.avg, auc

