import os
import shutil
from tqdm import tqdm
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import drn_binary as models
from helpers import count2


def remove_fake(args):
    """ Remove fake patches in the training set """
    datap = os.path.join(args.path_data, 'train/img1/')
    for f in tqdm(os.listdir(datap)):
        if f.startswith('trans'):
            shutil.move(os.path.join(datap, f), args.trans_patch_path + f)
    
    return


def aug_fake(args):
    """ Classify translated patches and add those classified as positive (unhealthy) into the training set
    
    params: threshold: pseudo labeling threshold for determining whether to add the translated patches
    """

    # firstly, remove all translated patches in training set back to the original translated_patches folder
    remove_fake(args)
            
    # before augment, count number of patches in each folder
    print('*********** Before adding fake patches ***********')
    count2(args.path_data)

    # move fake patches classified as unhealthy to the training set according to the aug_threshold
    move_translated(args)
    
    # count number of patches in each folder
    print('*********** After adding fake patches ***********')
    count2(args.path_data)

    return


def move_translated(args):
    """ move selected fake patches to the training set if them meet the threshold """
    
    model = models.__dict__[args.arch](args.pretrained)

    model = torch.nn.DataParallel(model).cuda()

    if args.aug_weights:
        if os.path.isfile(args.aug_weights):
            print("=> loading checkpoint/weights '{}'".format(args.aug_weights))
            checkpoint = torch.load(args.aug_weights)
            epo = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint/weights '{}' (epoch {})"
                  .format(args.aug_weights, epo))
        else:
            print("=> No checkpoint found at '{}'".format(args.aug_weights))
    else:
        print('=> No aug classification model provided. ')

    cudnn.benchmark = True
    #predit all translated patches
    model.eval()
    #number of samples predicted as 1
    n1 = 0
    #loop over all translated patches and predict class
    for f in tqdm(os.listdir(args.trans_patch_path)): 
        patch = os.path.join(args.trans_patch_path, f) 
        if not os.path.isdir(patch) and patch.lower().endswith(('.png', '.jpg', '.jpeg')):
            #print(patch) 
            img = Image.open(patch)
            
            with torch.no_grad():
                transform = transforms.Compose([
                            transforms.Resize(args.scale_size),
                            transforms.CenterCrop(args.crop_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
                image = torch.autograd.Variable(transform(img.convert('RGB')).unsqueeze(0).cuda())
                pred = model(image)
                pred_prob = F.softmax(pred)[0,1].item()
                if pred_prob >= args.aug_threshold:
                    dest = args.path_data + 'train/img1/' + f
                    shutil.move(patch, dest)
                    n1 += 1


    print('Number of fake unhealthy patches added to the training set: ', n1)

    return 