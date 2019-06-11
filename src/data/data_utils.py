#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image
import PIL.ImageChops
import os

import cv2
import glob
import numpy as np

def create_dict_texts(texts):

    texts = sorted(list(set(texts)))
    d = {l: i for i, l in enumerate(texts)}
    return d


def default_image_loader(path):
    
    img = Image.fromarray(cv2.resize(np.array(Image.open(path).convert('RGB')), (224, 224)))
    return img


def default_image_loader_tuberlin(path):

    img = Image.fromarray(cv2.resize(np.array(Image.open(path).convert('RGB')), (224, 224)))
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    return img

def default_sketch_loader_quickdraw(dir_sketch, cls_sketch, index):
    temp_ = np.load(os.path.join(dir_sketch, 'full_numpy_bitmap_' + cls_sketch + '.npy'))
    img = Image.fromarray(cv2.resize(temp_[index, :].reshape((28, 28)), (224, 224)))
    img = Image.merge("RGB", (img, img, img))
    img = PIL.ImageChops.invert(img)
    return img

def get_file_list(dir_skim, class_list, skim='sketch'):
    if skim=='sketch':
        _ext='*.png'
    elif skim=='images':
        _ext='*.jpg'
    else:
        NameError(skim + ' not implemented!')
        
    fnames = []
    fnames_cls = []
    for cls in class_list:
        path_file = glob.glob(os.path.join(dir_skim, cls, _ext))
        fnames += [os.path.basename(x) for x in path_file]
        fnames_cls += [cls]*len(path_file)
    return fnames, fnames_cls


def get_random_file_from_path(file_path):
    _ext='*.jpg'
    f_list = glob.glob(os.path.join(file_path, _ext))
    return np.random.choice(f_list,1)[0]


def get_file_list_quickdraw(dir_skim, class_list, skim='sketch'):
    if skim == 'sketch':
        _ext = '*.png'
    elif skim == 'images':
        _ext = '*.jpg'
    else:
        NameError(skim + ' not implemented!')
    no_of_sketch = 3000
    #fnames = []
    fnames_cls = []
    for i, cls in enumerate(class_list):
        temp_ = np.load(os.path.join(dir_skim, 'full_numpy_bitmap_' + cls + '.npy'))
        len_temp_ = temp_.shape[0]
        random_index = np.random.randint(len_temp_, size=no_of_sketch)
        if i==0:
            fnames = random_index
        else:
            fnames = np.concatenate((fnames, random_index), axis=None)
        #fnames.append(random_index)
        #path_file = glob.glob(os.path.join(dir_skim, cls, _ext))
        #fnames += [os.path.basename(x) for x in path_file]
        fnames_cls += [cls] * no_of_sketch
    return fnames, fnames_cls

def get_file_list_quickdraw_fixed(dir_skim, class_list, skim='sketch'):
    np.random.seed(42)
    no_of_sketch = 1000
    fnames_cls = []
    for i, cls in enumerate(class_list):
        temp_ = np.load(os.path.join(dir_skim, 'full_numpy_bitmap_' + cls + '.npy'))
        len_temp_ = temp_.shape[0]
        random_index = np.random.randint(len_temp_, size=no_of_sketch)
        if i==0:
            fnames = random_index
        else:
            fnames = np.concatenate((fnames, random_index), axis=None)
        fnames_cls += [cls] * no_of_sketch
    return fnames, fnames_cls

