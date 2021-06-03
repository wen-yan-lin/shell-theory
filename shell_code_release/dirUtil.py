import os
import shutil 
import glob
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import GlobalMaxPool2D



def dirLoadImages(dir_, types = ['*.jpg'], size = 224, interp = 'bilinear'): 
     # load images in a directory
    files = dirFiles(dir_, types)

    images = []
    for f in files:
        try:
            img = image.load_img(f, target_size=(size, size), interpolation='bilinear')
        except:
            print("%s is not an image. Skip." % (f))
            continue

        img_data = image.img_to_array(img)
        images.append(img_data)
    return images

def dirFiles(dir_, types_ = ['*']):  # get files in a directory
    files = []
    for ext  in  types_:
        files.extend(glob.glob(dir_ + "/" + ext, recursive=True))
    return sorted(files)


def dirFolder(dir_): # get subfolders of a directory
    x = next(os.walk(dir_))[1]
    for i in range(len(x)):
        x[i] = dir_ + x[i]
    return sorted(x)
    

def folder2feat(dir_, featType = 'resNet50', types = ['*.jpg', '*.JPG', '*.png'], size = 224, interp = 'bilinear'):
     # convert folder images to features

    model = Sequential()    
    if featType == 'vgg':
        vgg = VGG16(weights='imagenet', include_top=False)
        model.add(vgg)
    elif featType == 'resNet50':
        resNet = ResNet50(weights='imagenet', include_top=False)
        model.add(resNet)
        model.add(GlobalAvgPool2D())
    elif featType == 'mobileNet':
        mobileNet = MobileNet(weights='imagenet', include_top=False)
        model.add(mobileNet)
        model.add(GlobalAvgPool2D())

    else:
        print('no such feature')

    #model.summary()

    images = dirLoadImages(dir_, types, size, interp)

    features = []
    for im in images:
        img_data = np.expand_dims(im, axis=0)
        img_data = preprocess_input(img_data)
        feat = model.predict(img_data)
        features.append(feat)

    return features

def dir2feat(dir_, featType='resNet50'):
    # convert all subfolder folder images to features

    allFolders = dirFolder(dir_)
    allFeat = []
    total = 0
    for f in allFolders:
        print('Processing:' , f)
        feat = folder2feat(f, featType=featType)
        feat = np.concatenate(feat, axis=0)
        allFeat.append(feat)
        total = total + feat.shape[0]
        
    allGt = np.zeros(total, dtype=int)
    start = 0
    for i in range(len(allFeat)):
        allGt[start:start+allFeat[i].shape[0]] = i
        start = start + allFeat[i].shape[0]

    allFeat = np.concatenate(allFeat, axis=0)
    return allFeat, allGt
