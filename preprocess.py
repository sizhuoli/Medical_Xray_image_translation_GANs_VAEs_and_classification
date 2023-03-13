
import pandas as pd
import os
from tqdm import tqdm
from PIL import ImageOps
from PIL import Image
from shutil import copy
from shutil import copy2
import random

class data_preprocess():
    """ Data pre-process : crop original images and place in different folders according to their labels 
    
        Input : class configs containing path to the imgs and cropping options such as flip/equalization/binary/etc.
        Output : cropped patches in different folders; 
                 dataset for translation; 
                 dataset for classification.
    """
    def __init__(self, conf):
        self.meta = pd.read_csv(conf.path_to_meta)
        self.coor = pd.read_csv(conf.path_to_coor)
        self.oste = pd.read_csv(conf.path_to_oste)
        self.not_work = []
        self.imgs = {}
        
    def crop_1st(self, conf):
        """ Crop images in the foler named 'img/'"""
        #create subfolders
        folders = [conf.dest + 'img0', conf.dest + 'img1', conf.dest + 'img2', conf.dest + 'img3', conf.dest + 'img8', conf.dest + 'img9']
        for fd in folders:
            if not os.path.exists(fd):
                os.makedirs(fd)
        for filename in tqdm(os.listdir(conf.path_to_img1)):
            if filename.endswith('png'):
                try:
                    split = filename.find('_')
                    filter1 = self.coor["PATIENT_ID"]==int(filename[:split])
                    side = filename[split+1:-4]
                    filter2 = self.coor["SIDE"]==side
                    line = self.coor[filter1&filter2]
                    lat_coor = eval(line["lateral_coords"].iloc[0])
                    med_coor = eval(line["medial_coords"].iloc[0])
                    img = Image.open(conf.path_to_img1 + filename)
                    lat_cropped = crop(img, lat_coor, conf.size)
                    med_cropped = crop(img, med_coor, conf.size)
                    #find level (label)
                    f1 = self.oste['PATIENT_ID']==int(filename[:split])
                    f2 = self.oste['SIDE']==filename[split+1:-4]
                    f_lat = self.oste['compartment'] == 'lateral'
                    f_med = self.oste['compartment'] == 'medial'
                    lat_line = self.oste[f1&f2&f_lat]
                    med_line = self.oste[f1&f2&f_med]
                    lat_level = int(lat_line['osteophyte_tibia'].iloc[0]) #str
                    med_level = int(med_line['osteophyte_tibia'].iloc[0])
                    lat_folder = 'img'+str(lat_level)
                    med_folder = 'img'+str(med_level)
                    #mirror
                    lat_cropped, med_cropped = mirror(lat_cropped, med_cropped, side, conf.mirror)
                    #hist equalization
                    lat_cropped, med_cropped = equal(lat_cropped, med_cropped, conf.equal)
                    lat_cropped.save(conf.dest+lat_folder+"/"+"lat_"+filename)
                    med_cropped.save(conf.dest+med_folder+"/"+"med_"+filename)
                    self.imgs[filename] = None
                except:
                    self.not_work.append(filename)

    def crop_2nd(self, conf):
        """ Crop images in the foler 'img_2nd/'"""
        for filename in tqdm(os.listdir(conf.path_to_img2)):
            if filename.endswith('png') and filename not in self.imgs:

                try:
                    split = filename.find('_')
                    filter1 = self.meta["PATIENT_ID"]==filename[:split]
                    side = filename[split+1:-4]
                    filter2 = self.meta["SIDE"]== side
                    line = self.meta[filter1&filter2]
                    lat_coor = eval(line["coords_lat"].iloc[0])
                    med_coor = eval(line["coords_med"].iloc[0])
                    lat_level = int(line['OSTEOPHYTE_TIBIA_LATERAL'].iloc[0])
                    med_level = int(line['OSTEOPHYTE_TIBIA_MEDIAL'].iloc[0])
                    img = Image.open(conf.path_to_img2 + filename)
                    lat_cropped = crop(img, lat_coor, conf.size)
                    med_cropped = crop(img, med_coor, conf.size)
                    lat_folder = 'img'+str(lat_level)
                    med_folder = 'img'+str(med_level)
                    #mirror
                    lat_cropped, med_cropped = mirror(lat_cropped, med_cropped, side, conf.mirror)
                    #hist equalization
                    lat_cropped, med_cropped = equal(lat_cropped, med_cropped, conf.equal)
                    lat_cropped.save(conf.dest+lat_folder+"/"+"lat_"+filename)
                    med_cropped.save(conf.dest+med_folder+"/"+"med_"+filename)
                except:
                    self.not_work.append(filename)
                    if len(self.not_work) % 100 == 1:
                        print('Number of images that couldn"t be cropped:', len(self.not_work))
        
        print('**** Finished cropping.')
        #count files
        print('Number of patches in img0: ',len(os.listdir(conf.dest + 'img0')))
        print('Number of patches in img1: ',len(os.listdir(conf.dest + 'img1')))
        print('Number of patches in img2: ',len(os.listdir(conf.dest + 'img2')))
        print('Number of patches in img3: ',len(os.listdir(conf.dest + 'img3')))
        print('Number of images that couldn"t be cropped: ', len(self.not_work))
        
    
    
    def translation_split(self, conf):
        """ Create dataset for translation task :

        Split patches into subfolders: trainA (healthy)
                                       trainB (unhealthy)
                                       testA (healthy)
                                       testB (unhealthy)
        """
        trans_split(conf.dest, '/img0/', conf.trans_dataset, healthy = 1, train_ratio = conf.trans_ratio)
        if conf.only_sever == 1:
            print('Adding only label 3 patches to the "unhealthy" domain')
            #only sever
            trans_split(conf.dest, '/img3/', conf.trans_dataset, healthy = 0, train_ratio = conf.trans_ratio)
            #count files
            ntraina = len(os.listdir(os.path.join(conf.trans_dataset, 'trainA')))
            ntrainb = len(os.listdir(os.path.join(conf.trans_dataset, 'trainB')))
            ntesta = len(os.listdir(os.path.join(conf.trans_dataset, 'testA')))
            ntestb = len(os.listdir(os.path.join(conf.trans_dataset, 'testB')))
            print('Dataset creation finished.')
            print('Number of samples in trainA is {t0}\t'
                'Number of samples in trainB is {t1}\t'
                'Number of samples in testA is {te0}\t'
                'Number of samples in testB is {te1}\t'.format(t0 = ntraina, t1 = ntrainb, 
                                                                te0 = ntesta, te1 = ntestb))
        elif conf.only_sever == 0:
            print('**** Adding label 1, 2, 3 patches to the "unhealthy" domain ...')
            trans_split(conf.dest, '/img3/', conf.trans_dataset, healthy = 0, train_ratio = conf.trans_ratio)
            trans_split(conf.dest, '/img2/', conf.trans_dataset, healthy = 0, train_ratio = conf.trans_ratio)
            trans_split(conf.dest, '/img1/', conf.trans_dataset, healthy = 0, train_ratio = conf.trans_ratio)
            #count files
            ntraina = len(os.listdir(os.path.join(conf.trans_dataset, 'trainA')))
            ntrainb = len(os.listdir(os.path.join(conf.trans_dataset, 'trainB')))
            ntesta = len(os.listdir(os.path.join(conf.trans_dataset, 'testA')))
            ntestb = len(os.listdir(os.path.join(conf.trans_dataset, 'testB')))
            print('Dataset creation finished.')
            print('Number of samples in trainA is {t0}\t'
                'Number of samples in trainB is {t1}\t'
                'Number of samples in testA is {te0}\t'
                'Number of samples in testB is {te1}\t'.format(t0 = ntraina, t1 = ntrainb, 
                                                                te0 = ntesta, te1 = ntestb))
                                                                
                                                                
    def classification_split(self, conf):
        """ Create dataset for classification task :

        Split patches into subfolders: (if binary)
                                        train - img0 (healthy)
                                              - img1 (unhealthy)
                                        val   - img0 (healthy)
                                              - img1 (unhealthy)
                                        test  - img0 (healthy)
                                              - img1 (unhealthy)                       
        """
        if conf.binary:
            print('**** Creating binary dataset ...')
            classify_folders(conf.dest + 'img0', conf.class_dataset, 'img0', ratio = conf.classify_ratio)
            classify_folders(conf.dest + 'img1', conf.class_dataset, 'img1', ratio = conf.classify_ratio)
            classify_folders(conf.dest + 'img2', conf.class_dataset, 'img1', ratio = conf.classify_ratio)
            classify_folders(conf.dest + 'img3', conf.class_dataset, 'img1', ratio = conf.classify_ratio)
            #count files
            ntrain0 = len(os.listdir(os.path.join(conf.class_dataset, 'train/img0')))
            ntrain1 = len(os.listdir(os.path.join(conf.class_dataset, 'train/img1')))
            ntest0 = len(os.listdir(os.path.join(conf.class_dataset, 'test/img0')))
            ntest1 = len(os.listdir(os.path.join(conf.class_dataset, 'test/img1')))
            nval0 = len(os.listdir(os.path.join(conf.class_dataset, 'val/img0')))
            nval1 = len(os.listdir(os.path.join(conf.class_dataset, 'val/img1')))
            print('Dataset creation finished.')
            print('Number of samples in train_0 is {t0}\t'
                'Number of samples in train_1 is {t1}\t'
                'Number of samples in val_0 is {v0}\t'
                'Number of samples in val_1 is {v1}\t'
                'Number of samples in test_0 is {te0}\t'
                'Number of samples in test_1 is {te1}\t'.format(t0 = ntrain0, t1 = ntrain1, 
                                                                v0 = nval0, v1 = nval1, 
                                                                te0 = ntest0, te1 = ntest1))
        else:
            print('**** Creating multi-class dataset')
            classify_folders(conf.dest + 'img0', conf.class_dataset, 'img0', ratio = conf.classify_ratio)
            classify_folders(conf.dest + 'img1', conf.class_dataset, 'img1', ratio = conf.classify_ratio)
            classify_folders(conf.dest + 'img2', conf.class_dataset, 'img2', ratio = conf.classify_ratio)
            classify_folders(conf.dest + 'img3', conf.class_dataset, 'img3', ratio = conf.classify_ratio)
            print('Dataset creation finished.')
            #count files
            ntrain0 = len(os.listdir(os.path.join(conf.class_dataset, 'train/img0')))
            ntrain1 = len(os.listdir(os.path.join(conf.class_dataset, 'train/img1')))
            ntrain2 = len(os.listdir(os.path.join(conf.class_dataset, 'train/img2')))
            ntrain3 = len(os.listdir(os.path.join(conf.class_dataset, 'train/img3')))
            ntest0 = len(os.listdir(os.path.join(conf.class_dataset, 'test/img0')))
            ntest1 = len(os.listdir(os.path.join(conf.class_dataset, 'test/img1')))
            ntest2 = len(os.listdir(os.path.join(conf.class_dataset, 'test/img2')))
            ntest3 = len(os.listdir(os.path.join(conf.class_dataset, 'test/img3')))
            nval0 = len(os.listdir(os.path.join(conf.class_dataset, 'val/img0')))
            nval1 = len(os.listdir(os.path.join(conf.class_dataset, 'val/img1')))
            nval2 = len(os.listdir(os.path.join(conf.class_dataset, 'val/img2')))
            nval3 = len(os.listdir(os.path.join(conf.class_dataset, 'val/img3')))
            print('Number of samples in train_0 is {t0}\t'
                'Number of samples in train_1 is {t1}\t'
                'Number of samples in train_2 is {t2}\t'
                'Number of samples in train_3 is {t3}\n'
                'Number of samples in val_0 is {v0}\t'
                'Number of samples in val_1 is {v1}\t'
                'Number of samples in val_2 is {v2}\t'
                'Number of samples in val_3 is {v3}\n'
                'Number of samples in test_0 is {te0}\t'
                'Number of samples in test_1 is {te1}\t'
                'Number of samples in test_2 is {te2}\t'
                'Number of samples in test_3 is {te3}\t'.format(t0 = ntrain0, t1 = ntrain1, t2 = ntrain2, t3 = ntrain3,  
                                                                v0 = nval0, v1 = nval1, v2 = nval2, v3 = nval3,
                                                                te0 = ntest0, te1 = ntest1, te2 = ntest2, te3 = ntest3))



def mirror(lat_cropped, med_cropped, side, mirror):
    """ Flip cropped patches --> all patches on the same side """
    if mirror == 1:
        if side == 'al':
            lat_cropped = ImageOps.mirror(lat_cropped)
        else:
            med_cropped = ImageOps.mirror(med_cropped)
    return lat_cropped, med_cropped



def equal(lat_cropped, med_cropped, equal):
    """ Apply histogram equalization to the cropped patches """
    if equal == 1:
        lat_cropped = ImageOps.equalize(lat_cropped)
        med_cropped = ImageOps.equalize(med_cropped)
    return lat_cropped, med_cropped



def crop(img,coord, size):
    """ Crop patches according to coordinates and cropping size """
    width, high = img.size
    if coord[0]-size/2 >0 and coord[0]+size/2 < width:
        cropped = img.crop((coord[0]-size/2, coord[1]-size/2,coord[0]+size/2 , coord[1]+size/2))  # (left, upper, right, lower)
    elif coord[0]-size/2 <=0:
        cropped = img.crop((0, coord[1]-size/2,size , coord[1]+size/2))  # (left, upper, right, lower)
    elif coord[0]+size/2 >= width:
        cropped = img.crop((width-size, coord[1]-size/2, width, coord[1]+size/2))  # (left, upper, right, lower)
    return cropped



def trans_split(base, path, dest, healthy = 1, train_ratio = 0.9):
    """ Split for translation task 
    
    params: base: path to the patches
            path: subfolder for patches
            dest: path to the dataset
            healthy: if healthy or unhealthy
            train_ratio: ratio for training set, (1-ratio) for testing set
    """
    folder = os.listdir(base + path)
    num = len(folder)
    # print( "number of images ", num)
    index_list = list(range(num))
    random.shuffle(index_list)
    c = 0
    if healthy == 1:
        train = dest + '/trainA'
        test = dest + '/testA'
    else:
        train = dest + '/trainB'
        test = dest + '/testB'
    if not os.path.exists(train) or not os.path.exists(test):
        os.makedirs(train)
        os.makedirs(test)
    for i in tqdm(index_list):
        filename = os.path.join(base + path, folder[i])
        if c < num*train_ratio:
            copy(filename, train)
        else:
            copy(filename, test)
        c += 1
    return 



def classify_folders(dire, path_data, dest, ratio = (0.8, 0.1, 0.1)):
    """ Split for classification task 
    
    params: dire: path to the cropped patches
            path_data: path to the dataset
            dest: subfolder in the dataset
    """
    base = os.listdir(dire)
    num = len(base)
    index_list = list(range(num))
    random.shuffle(index_list)
    c = 0
    train = os.path.join(path_data + 'train/', dest)
    val = os.path.join(path_data + 'val/', dest)
    test = os.path.join(path_data + 'test/', dest)
    if not os.path.exists(train):
        os.makedirs(train)
    if not os.path.exists(val):
        os.makedirs(val)
    if not os.path.exists(test):
        os.makedirs(test)
    for i in tqdm(index_list):
        fileName = os.path.join(dire, base[i])
        if c < num*ratio[0]:
            copy2(fileName, train)
        elif c > num*(ratio[0]+ratio[1]):
            copy2(fileName, test)
        else:
            copy2(fileName, val)
        c += 1
    return 