#code reference: Fisher Yu and Eric Smith, https://github.com/fyu/drn
#code reference: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py



import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_files(dire):
    """ Count files in a folder"""
    return len(os.listdir(dire))

def count2(base):
    """ Count files in the train/val/test sets"""
    train_0 = os.path.join(base, 'train/img0')
    train_1 = os.path.join(base, 'train/img1')
    val_0 = os.path.join(base, 'val/img0')
    val_1 = os.path.join(base, 'val/img1')
    test_0 = os.path.join(base, 'test/img0')
    test_1 = os.path.join(base, 'test/img1')
    print('Number of samples in train_0 is {t0}\t'
    'Number of samples in train_1 is {t1}\n'
    'Number of samples in val_0 is {v0}\t'
    'Number of samples in val_1 is {v1}\n'
    'Number of samples in test_0 is {te0}\t'
    'Number of samples in test_1 is {te1}\n'.format(t0 = count_files(train_0), t1 = count_files(train_1), 
                                                    v0 = count_files(val_0), v1 = count_files(val_1), 
                                                    te0 = count_files(test_0), te1 = count_files(test_1)))


class FocalLoss(nn.Module):
    """
    Focal loss
    """
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
    
def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    """ Save weights/checkpoints """
    torch.save(state, filename)
    if is_best:
        shutil.copy2(filename, args.savepath + 'model_best.pth.tar')


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
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

def adjust_learning_rate(args, optimizer, epoch):
    """ Sets the learning rate to the initial LR decayed by 10 every 30 epochs """
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    print('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_metrics(output, target, eps = 1e-7):
    """ Compute metrics during training 
    
    sensivity: TP / (TP + FN)

    specificity: TN / (TN + FP)

    precision: TP / (TP + FP)

    f1: 2TP / (2TP + FP + FN)
    """
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()[0]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(batch_size):
        #print(pred[i], target[i])
        if pred[i] == 1 and target[i] == 1:
            tp += 1
        elif pred[i] == 0 and target[i] == 0:
            tn += 1
        elif pred[i] == 0 and target[i] == 1:
            fn += 1
        elif pred[i] == 1 and target[i] == 0:
            fp += 1
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn)
    spec = tn / (tn + fp + eps)
    return recall, spec, f1, acc
