from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from scipy import stats
from tqdm import tqdm
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F


import drn_binary as models
from helpers import AverageMeter, FocalLoss, compute_metrics



def evaluate_predict(args):
    """ Threshold tuning by evaluating on the validation set. 
        
        Output: labels and predicted probabilities.
    """
    # create model
    model = models.__dict__[args.arch](args.pretrained)

    model = torch.nn.DataParallel(model).cuda()

    if args.load_weights:
        if os.path.isfile(args.load_weights):
            print("=> loading checkpoint '{}'".format(args.load_weights))
            checkpoint = torch.load(args.load_weights)
            epo = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_weights, epo))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_weights))

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.path_data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # save labels and predicted probabilities for computing roc auc and bootstrap resampling
    y_label = []
    y_pred_prob = []

    t = transforms.Compose([
        transforms.Resize(args.scale_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, t),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    criterion = FocalLoss().cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    sens = AverageMeter()
    spec = AverageMeter()
    f1s = AverageMeter()
    accs = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        pred = model(input_var)
        # prediction probabilities
        pred_prob = F.softmax(pred)
        y_label.extend(list(target.cpu().numpy()))
        y_pred_prob.extend(list(pred_prob.cpu().detach().numpy()[:,1]))

        loss = criterion(pred, target_var)

        # compute metrics and record loss
        sensitivity, specificity, f1, acc = compute_metrics(pred.data, target)
        losses.update(loss.data.item(), input.size(0))
        sens.update(sensitivity, input.size(0))
        spec.update(specificity, input.size(0))
        f1s.update(f1, input.size(0))
        accs.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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
    return y_label, y_pred_prob



def sen_spe_curve(labels, probs, args):
    """ Find the best threshold t so that sensitivity and specificity are close enough.
        Plot sens/spec curve for visualization (saved in results folder).
        
        Output: best threshold.
    """
    probs_ar = np.array(probs)
    # draw sen/spe curve
    # for 50 different t values in the interval (0.3, 0.8)
    tl = list(np.linspace(0.3, 0.8, num=50))
    sensl = []
    specl = []
    f1l = []
    for i in range(len(tl)):
        ti = tl[i]
        preds=np.zeros(probs_ar.shape)
        preds[probs_ar>ti]=1
        cm = confusion_matrix(labels,preds)
        sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
        specificity = cm[0,0]/(cm[0,0]+cm[0,1])
        precision = cm[1,1]/(cm[1,1] + cm[0,1])
        f1 = 2*precision*sensitivity/(precision + sensitivity)
        
        sensl.append(sensitivity)
        specl.append(specificity)
        f1l.append(f1)
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.plot(tl,sensl, label = 'Sens.')
    ax.plot(tl, specl, label = 'Spec.')
    ax.plot(tl, f1l, label = 'F1')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEE')
    ax.xaxis.grid(False)
    plt.title('Sensitivity / Specificity / F1 Curve', fontsize = 14, pad=15,color='#333333',weight = 'bold')
    plt.legend()
    plt.xlabel('Decision Threshold', fontsize = 12, labelpad=15,color='#333333')
    plt.ylabel('Value', fontsize = 12, labelpad=15,color='#333333')
    plt.savefig(args.results_path + args.model_name + '_sens_spec_f1_curve' + '.png')
    plt.show()

    # find best threshold so that sens and spec have smallest difference
    # init threshold 0.5
    thres = 0.5
    # min difference
    mindf = 1
    for i in range(len(tl)):
        if abs(sensl[i]-specl[i]) < mindf:
            mindf = abs(sensl[i]-specl[i])
            thres = tl[i]

    return thres   # best threshold



def bootstrap(y_label, y_prob, t, model_name, itr, alpha):
    """ Bootstrap resampling (percentil method). """
    sen_list = []
    spe_list = []
    f1_list = []
    auc_list = []
    for i in range(itr):
        idx = np.random.choice(len(y_label), len(y_label))

        sam_label = np.array(y_label)[idx]
        sam_prob = np.array(y_prob)[idx]
        
        preds=np.zeros(sam_prob.shape)
        preds[sam_prob>t]=1

        cm = confusion_matrix(sam_label,preds)
        total1=sum(sum(cm))
        acc=(cm[0,0]+cm[1,1])/total1
        sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
        specificity = cm[0,0]/(cm[0,0]+cm[0,1])
        precision = cm[1,1]/(cm[1,1] + cm[0,1])

        auc = roc_auc_score(sam_label, sam_prob)

        f1 = 2*precision*sensitivity/(precision + sensitivity)
        sen_list.append(sensitivity)
        spe_list.append(specificity)
        f1_list.append(f1)
        auc_list.append(auc)

    sen_ar = np.array(sen_list)
    spe_ar = np.array(spe_list)
    f1_ar = np.array(f1_list)
    auc_ar = np.array(auc_list)
    print('Model: ', model_name)
    print('-------------------------------------')
    print('Sensitivity: ')
    stat(sen_ar, itr, alpha)
    print('-------------------------------------')
    print('Specificity: ')
    stat(spe_ar, itr, alpha)
    print('-------------------------------------')
    print('f1: ')
    stat(f1_ar, itr, alpha)
    print('-------------------------------------')
    print('auc: ')
    stat(auc_ar, itr, alpha)
    print('-------------------------------------')
    return 


def stat(array, itr, alpha):
    """ Compute statistics for given array """
    mean = np.mean(array)
    sort_l = np.sort(array)
    lower = sort_l[int(itr*alpha/2)]
    upper = sort_l[int(itr*(1-alpha/2))]
    print('Mean: %.3f' % mean)
    print('Confidence Interval: [{low:.3f}, {up:.3f}]'.format(low = lower, up = upper))
    return 


def threshold_tuning(args):
    """ Find best threshold so that sensitivity and specificity are close enough.
        Print bootstrap results for sens/spec/f1/auc on the validation set.

    Params: Configs of a model

    Outputs: best threshold
             bootstrap results on the validation set when using default threshold and the best threshold

    """
    # path to save result images
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    labels, probs = evaluate_predict(args)
    best_t = sen_spe_curve(labels, probs, args)
    print('*****************************************************************************')
    print(' Bootstrap results on the validation set using the best threshold of %.3f' % best_t)
    bootstrap(labels, probs, best_t, args.model_name, args.bootstrap_itr, args.bootstrap_alpha)

    print('*****************************************************************************')
    print(' Bootstrap results on the validation set using the default threshold of 0.5')
    bootstrap(labels, probs, 0.5, args.model_name, args.bootstrap_itr, args.bootstrap_alpha)

    return round(best_t, 3)






class test_config:
    """ Setup configs when testing on the test data. """
    path_data = 'data/'
    arch = 'drn_c_26'
    workers = 4
    batch_size = 32
    print_freq = 10
    pretrained = 0
    crop_size = 224
    scale_size = 256
    def __init__(self, weights):
        self.load_weights = weights


def test_predict(args):
    """ Evaluation on the test set.
        
        Output: labels and predicted probabilities of the test patches.    
    """

    model = models.__dict__[args.arch](args.pretrained)

    model = torch.nn.DataParallel(model).cuda()

    if args.load_weights:
        if os.path.isfile(args.load_weights):
            print("=> loading checkpoint '{}'".format(args.load_weights))
            checkpoint = torch.load(args.load_weights)
            epo = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_weights, epo))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_weights))


    cudnn.benchmark = True

    # Data loading code
    testdir = os.path.join(args.path_data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    y_label = []
    y_pred_prob = []

    t = transforms.Compose([
        transforms.Resize(args.scale_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, t),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = FocalLoss().cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        pred = model(input_var)
        pred_prob = F.softmax(pred)
        y_label.extend(list(target.cpu().numpy()))
        y_pred_prob.extend(list(pred_prob.cpu().detach().numpy()[:,1]))

        loss = criterion(pred, target_var)

        sensitivity, specificity, f1, acc = compute_metrics(pred.data, target)
        losses.update(loss.data.item(), input.size(0))
        accs.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses, acc = accs))
            
    #compute roc auc
    auc = roc_auc_score(y_label, y_pred_prob)
    
    print(' * Acc {accs.avg:.3f} ROC AUC {auc:.3f}'.format(accs = accs, auc = auc))
    
    return y_label, y_pred_prob



def test_bootstrap(weight_list, t_list, num_metrics, itr):
    """ Bootstrap resampling on the test data. 
        Compute for all models in one run => all models are tested on the same resampled data
    """
    
    # array to store predictions during resampling
    results = np.zeros((len(weight_list), num_metrics, itr))
    # count the number of patches in the test data
    test_conf1 = test_config(weight_list[0])
    test0 = len(os.listdir(os.path.join(test_conf1.path_data, 'test/img0/')))
    test1 = len(os.listdir(os.path.join(test_conf1.path_data, 'test/img1/')))
    num_test_data = test0 + test1
    # 1887
    print('Number of test patches: ', num_test_data)
    
    # array to store predicted probs
    prob_array = np.zeros((num_test_data, len(weight_list)))
    c = 0
    for w in weight_list:
        test_conf = test_config(w)
        labels, probs = test_predict(test_conf)
        prob_array[:, c] = np.array(probs)
        c+=1
    labels = np.array(labels)
    
    # do resampling for itr times
    for i in tqdm(range(itr)):
        
        # randomly sample (number of test patches) patches from the test set
        idx = np.random.choice(num_test_data, num_test_data)
        #for each column
        for col in range(prob_array.shape[1]):
            prob_ar = prob_array[:, col]
            t = t_list[col]
            results[col, :, i] = compute_boots(prob_ar, labels, idx, t)
    
    return results

def compute_boots(prob_ar, label, idx, t):
    """ Compute bootstrap results given probs, labels, resampling indices idx and threshold t. """
    
    # pick patches according to the resampling indices
    sam_label = label[idx]
    sam_prob = prob_ar[idx]
    
    preds=np.zeros(sam_prob.shape)
    preds[sam_prob>t]=1

    #tn, fp, fn, tp = confusion_matrix(sam_label,preds).ravel()
    cm = confusion_matrix(sam_label,preds)
    #print('Confusion Matrix : \n', cm)
    total1=sum(sum(cm))
    acc=(cm[0,0]+cm[1,1])/total1
    #print ('Accuracy : ', acc)
    sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
    #print('Sensitivity : ', sensitivity)
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    #print('Specificity : ', specificity)
    precision = cm[1,1]/(cm[1,1] + cm[0,1])
    auc = roc_auc_score(sam_label, sam_prob)
    f1 = 2*precision*sensitivity/(precision + sensitivity)

    return sensitivity, specificity, auc, f1

   

def test_bootstrap_results(results, metrics, model_names, itr, alpha):
    """ Print bootstrap results for all models"""

    # for each model
    for i in range(results.shape[0]):
        # for each metric
        for j in range(results.shape[1]):
            print('Model: ', model_names[i])
            print('Metric: ', metrics[j])
            vals = results[i,j,:]
            vals_sorted = np.sort(vals)
            mu = np.mean(vals_sorted)
            print('*******************')
            print('Mean: %.3f'% mu)
            # get lower and upper values (confidence interval)
            lower = vals_sorted[int(itr*alpha/2)]
            upper = vals_sorted[int(itr*(1-alpha/2))]
            print('Confidence Interval: [{low:.3f}, {up:.3f}]'.format(low = lower, up = upper))
            print('----------------------------------------------')
            print('----------------------------------------------')

    return


def test_ttest_auc(results, model_names, results_path):
    """ T test for roc auc """
    # roc auc results
    aucs = results[:, 2, :]
    # baseline auc
    base_auc = aucs[0]
    #normal dis test
    t_nor, p_nor = stats.shapiro(base_auc)
    print('Normal test for baseline model: ')
    print('p_normal value: ', p_nor)
    # itertively for each new model, plot auc distribution and compare with the baseline model
    for i in range(1, results.shape[0]):
        auci = aucs[i]
        #normal test
        t_nor, p_nor = stats.shapiro(auci)
        print('Normal test: ')
        print('p_normal value: ', p_nor)
        # t test
        t, p = stats.ttest_ind(auci,base_auc, equal_var = False)
        print('Signficant test of two means: ')
        print('t, p value of model baseline compared with model', model_names[i])
        print('t:', t)
        print('p:', p/2)
        print('************************************************************')
        # plot distribution
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.hist(auci, bins = 100, facecolor = '#FF7C7C', alpha = 0.7, label = model_names[i])
        ax.hist(base_auc, bins = 100, facecolor = '#60ACFC', alpha = 0.7, label = 'baseline')
        plt.legend()
        plt.xlabel('ROC AUC Score', fontsize = 12, labelpad=15,color='#333333')
        plt.ylabel('Count', fontsize = 12, labelpad=15,color='#333333')
        plt.title('Histograms of ROC AUC (Baseline VS. New Model)',fontsize = 14,pad=15,color='#333333',weight ='bold')
        plt.savefig(results_path + 'auc_hist_' + model_names[i] + '.png')
        plt.show()

    return

def test_ttest_f1(results, model_names, results_path):
    """ T test for f1 """
    # f1 results 
    f1s = results[:, 3, :]
    # baseline f1
    base_f1 = f1s[0]
    #normal dis test
    t_nor, p_nor = stats.shapiro(base_f1)
    print('Normal test for baseline model: ')
    print('p_normal value: ', p_nor)
    # itertively for each new model, plot auc distribution and compare with the baseline model
    for i in range(1, results.shape[0]):
        f1i = f1s[i]
        #normal test
        t_nor, p_nor = stats.shapiro(f1i)
        print('Normal test: ')
        print('p_normal value: ', p_nor)
        # t test
        t, p = stats.ttest_ind(f1i,base_f1, equal_var = False)
        print('Signficant test of two means: ')
        print('t, p value of model baseline compared with model', model_names[i])
        print('t:', t)
        print('p:', p/2)
        print('************************************************************')
        # plot distribution
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.hist(f1i, bins = 100, facecolor = '#FF7C7C', alpha = 0.7, label = model_names[i])
        ax.hist(base_f1, bins = 100, facecolor = '#60ACFC', alpha = 0.7, label = 'baseline')
        plt.legend()
        plt.xlabel('F1 Score', fontsize = 12, labelpad=15,color='#333333')
        plt.ylabel('Count', fontsize = 12, labelpad=15,color='#333333')
        plt.title('Histograms of F1 (Baseline VS. New Model)', fontsize = 14, pad=15,color='#333333',weight = 'bold')
        plt.savefig(results_path + 'f1_hist_' + model_names[i] + '.png')
        plt.show()

    return


class test_evaluations:
    """ Bootstrap resampling and t test on the test data"""
    
    def __init__(self, weight_list, t_list, metrics, model_names, results_path, itr, alpha):
        self.weight_list = weight_list
        self.t_list = t_list
        self.metrics = metrics
        self.model_names = model_names
        self.results_path = results_path
        self.itr = itr
        self.alpha = alpha
        self.res = test_bootstrap(self.weight_list, self.t_list, len(self.metrics), self.itr)

    def bootstrap(self):
        test_bootstrap_results(self.res, self.metrics, self.model_names, self.itr, self.alpha)

    def t_test_auc(self):
        test_ttest_auc(self.res, self.model_names, self.results_path)
        
    def t_test_f1(self):
        test_ttest_f1(self.res, self.model_names, self.results_path)
