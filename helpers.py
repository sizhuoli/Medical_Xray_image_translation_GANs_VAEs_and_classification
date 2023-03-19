#code reference: Ming-Yu Liu and Christian Clauss and et al, https://github.com/mingyuliutw/UNIT


from torch.utils import data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
import torchvision.utils as vutils
import torch.nn.init as init
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import math
import yaml
import numpy as np
import time
try:
    from itertools import izip as zip
except ImportError:
    pass
from PIL import Image
import os




IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def load_config(config):
    """ Load configs from yaml file. """
    with open(config, 'r') as stream:
        return yaml.load(stream)


def data_loaders(conf, data_root):
    """ Dataloader for translation task. """
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size_a = new_size_b = conf['new_size']
    
    # loader training data from domain a (healthy)
    train_loader_a = loader_folder(os.path.join(data_root, 'trainA'), batch_size, True,
                                            new_size_a, num_workers)
    # loader test data from domain a (healthy)
    test_loader_a = loader_folder(os.path.join(data_root, 'testA'), batch_size, False,
                                            new_size_a, num_workers)
    # loader training data from domain b (unhealthy)
    train_loader_b = loader_folder(os.path.join(data_root, 'trainB'), batch_size, True,
                                            new_size_b, num_workers)
    # loader test data from domain b (unhealthy)
    test_loader_b = loader_folder(os.path.join(data_root, 'testB'), batch_size, False,
                                            new_size_b, num_workers)
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def loader_folder(input_folder, batch_size, train, new_size, num_workers):
    """ Load data from folders. """
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    dataset = ImageFolder(input_folder, transform=transform)
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def __display_images1(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def display_images(image_outputs, display_image_num, image_directory, postfix):
    """ Display images during training. """
    n = len(image_outputs)
    __display_images1(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __display_images1(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))


def output_sub_folders(output_directory):
    """ Create subfolders for images and checkpoints. """
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory



def write_loss(iterations, trainer, train_writer):
    """ Write loss in tensorboard. """
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    """ Learning rate scheduler. """
    scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    return scheduler


def weights_init(init_type='gaussian'):
    """ Weight init methods. """
    def init_fun(m):
        classname = m.__class__.__name__
        #for each conv or linear layer in the model which has the attribute named 'weight'
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

class Timer:
    """ Time counter. """
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def default_loader(path):
    """ Convert image to RGB. """
    return Image.open(path).convert('RGB')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class ImageFolder(data.Dataset):
    """ Image folder for translation task. 
        
        Data is stored in the following format:
        
        - data/
               trainA/
               trainB/
               testA/
               testB/
    """
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)