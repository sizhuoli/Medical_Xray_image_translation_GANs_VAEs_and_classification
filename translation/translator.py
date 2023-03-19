
from helpers import load_config
import os
from tqdm import tqdm
from unit_trainer import UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from skimage import color
import cv2



class PatchTranslator:
    """ Translate patches from healthy to unhealthy

    params: conf: load config of the model
            weights: load weights
            patch_path: healthy patches to translate (h --> uh)
            save_path: path to save translated patches
    """
    def __init__(self, opts):
        self.conf = opts.config_file
        self.weights = opts.weights
        self.patch_path = opts.patch_path
        self.save_path = opts.save_path

    def translate(self):
        cudnn.benchmark = True
        config = load_config(self.conf)
        trainer = UNIT_Trainer(config)
        state_dict = torch.load(self.weights)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
        trainer.cuda()
        trainer.eval()

        #a to b (healthy --> unhealthy)
        encode = trainer.gen_a.encode 
        #generator encoder
        style_encode = trainer.gen_b.encode 
        decode = trainer.gen_b.decode
        resize = config['new_size']
        #iterate over all patches
        for f in tqdm(os.listdir(self.patch_path)): 
            patch = os.path.join(self.patch_path, f) 
            # for each patch
            if not os.path.isdir(patch) and patch.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(patch)
                with torch.no_grad():
                    transform = transforms.Compose([transforms.Resize(resize),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                    image = Variable(transform(img.convert('RGB')).unsqueeze(0).cuda())

                    #translation
                    content, _ = encode(image)
                    output = decode(content)

                    #rescale to 0-255 and resize to original
                    output = (output + 1) * 255 / 2. 
                    output = F.interpolate(output, size=180)
                    output = output.cpu()[0].permute(1,2,0)
                    output = color.rgb2gray(np.asarray(output))

                    #save translated patches
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    filepath = self.save_path + 'trans_' + f
                    cv2.imwrite(filepath, output)
        
        print('*** Translation finished.')
        print(' Number of translate patches: ', len(os.listdir(self.patch_path)))