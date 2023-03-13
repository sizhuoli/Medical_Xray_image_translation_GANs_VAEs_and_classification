# ------- Medical Image Synthesis Using VAEs & GANs ------- 


## This project contains: 
1. Preprocessing of X-ray images of knee joint
2. Training of a VAE + GAN network for image2image translation
3. Training of DRNs for binary classification
4. Verification of the generated fake images


## Software dependency:
- Python3
- torch
- torchvision
- PIL
- tensorboard


## Running steps:
### 1. Data pre-processing
    -----------------------------------------------------------
    Working directory: 'final_files/raw_data/'
    Original data: 'img/' & 'img_2nd/'
    -----------------------------------------------------------
    Run 'data_preprocess.ipynb'
    ==> Crop images
    ==> Create datasets for translation and classificiation tasks
    ==> Cropped patches saved in 'cropped_patches/'

### 2. Training of translation model 
    -----------------------------------------------------------
    Working directory: 'final_files/translation/'
    Data: 'data/'
    -----------------------------------------------------------
    Run 'gan_training.ipynb'
    ==> Train GAN network
    ==> Trained weights and sample images saved in 'data/outputs/'

### 3. Generation of fake unhealthy patches
    -----------------------------------------------------------
    Working directory: 'final_files/translation/'
    Data: 'data/'
    -----------------------------------------------------------
    Run 'translate_patches.ipynb'
    ==> Translate healthy patches into unhealthy ones
    ==> Translated patches saved in 'translated_patches/'

### 4. Training of baseline classification model
    -----------------------------------------------------------
    Working directory: 'final_files/classification/'
    Data: 'data/'
    -----------------------------------------------------------
    Run 'classification_training.ipynb' (STEP 0 & 1)
    ==> Trained model saved in 'classify_01_models/'

### 5. Training of new classification models with augmented dataset
    -----------------------------------------------------------
    Working directory: 'final_files/classification/'
    Data: 'data/'
    -----------------------------------------------------------
    Run 'classification_training.ipynb' (STEP 0 & 2)
    ==> Set different thresholds in configs to train models with different training data
        (e.g. 
        if aug_threshold = 0.9:
            add fake patches with a predicted probability > 0.9
            to the unhealthy domain in the original training dataset)
    ==> Trained model saved in 'classify_01_models/'

### 6. Evaluation
    -----------------------------------------------------------
    Working directory: 'final_files/classification/'
    Data: 'data/'
    -----------------------------------------------------------
    1). Threshold tuning on the validation data
    Run 'model_evaluation.ipynb' (STEP 0 & 1)
    ==> Find best threshold for each model so that sensitivity and specificity are similar
    ==> Print bootstrap results on the validation set 
        using the best threshold and default threshold 0.5

    2). Evaluation on the test data
    Run 'model_evaluation.ipynb' (STEP 2)
    ==> Print bootstrap results on the test set 
        using best thresholds obtained from STEP 1 and default threshold 0.5
    ==> T test for ROC AUC scores:
        Plot histograms of AUC (compared with the baseline)
        Compute t and p values
    ==> Plots saved in 'results/'


## Code reference *

https://github.com/fyu/drn
###
https://github.com/mingyuliutw/UNIT
#
## Industrial collaborator
Radiobotics ApS, Denmark 
#
## Master's thesis
Author: Sizhuo Li\\
Supervisors: Bulat Ibragimov and Eric Navarro (Radiobotics)\\
University of Copenhagen\\
August, 2020\\
