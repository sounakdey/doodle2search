#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data

import os
import pickle
import random
import sys

sys.path.insert(0, '.')
from data.class_word2vec import create_class_embeddings

import numpy as np
from scipy.spatial.distance import cdist
from data.data_utils import *


def QuickDraw_Extended(args, transform='None'):
    # Getting the classes
    class_labels_file = os.path.join(args.data_path, 'quick_draw_125', 'class_labels.txt')
    list_class = open(class_labels_file).read().splitlines()
    dicts_class = create_dict_texts(list_class)

    # Random seed
    np.random.seed(args.seed)

    # Create the class embeddings
    if os.path.isfile('./data/semantic_labels_sketchy.npy'):
        class_emb = np.load('./data/semantic_labels_quickdraw.npy')
        with open("./data/vocab_quickdraw.pkl", "rb") as input_file:
            vocab = pickle.load(input_file)
    else:
        class_emb = create_class_embeddings(list_class, args.dataset)
        vocab = list_class

    # Read test classes
    with open("../data/zeroshot_classes_quickdraw.txt") as fp:
        test_class = fp.read().splitlines()

    list_class = [x for x in list_class if x not in test_class]
    # Random Shuffle
    random.seed(args.seed)
    shuffled_list_class = list_class
    random.shuffle(shuffled_list_class)

    # Dividing the classes
    train_class = shuffled_list_class[:int(0.9 * len(shuffled_list_class))]
    valid_class = shuffled_list_class[int(0.9 * len(shuffled_list_class)):]

    if args.exp_idf is not None:
        if args.save is None:
            args.save = os.path.join('./checkpoint', args.exp_idf)
        with open(os.path.join(args.save, 'train.txt'), 'w') as fp:
            for item in train_class:
                fp.write("%s\n" % item)
        with open(os.path.join(args.save, 'valid.txt'), 'w') as fp:
            for item in valid_class:
                fp.write("%s\n" % item)

        if args.plot is False:
            with open(os.path.join(args.save, 'valid.txt'), 'r') as fp:
                valid_class = fp.read().splitlines()

    # Data Loaders
    train_loader =QuickDraw_Extended_train(args, train_class, dicts_class, class_emb, vocab, transform)
    valid_sk_loader = QuickDraw_Extended_valid_test(args, valid_class, dicts_class, class_emb, vocab, transform,
                                                  type_skim='sketch')
    valid_im_loader = QuickDraw_Extended_valid_test(args, valid_class, dicts_class, class_emb, vocab, transform,
                                                  type_skim='images')
    test_sk_loader = QuickDraw_Extended_valid_test(args, test_class, dicts_class, class_emb, vocab, transform,
                                                 type_skim='sketch')
    test_im_loader = QuickDraw_Extended_valid_test(args, test_class, dicts_class, class_emb, vocab, transform,
                                                 type_skim='images')

    return train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dicts_class


class QuickDraw_Extended_valid_test(data.Dataset):
    def __init__(self, args, set_class, dicts_class, class_emb, vocab, transform=None, type_skim='images'):
        self.transform = transform
        self.plot = args.plot
        self.set_class = set_class
        self.dicts_class = dicts_class
        self.word2vec = class_emb
        self.vocab = vocab

        if type_skim == 'images':
            self.dir_file = os.path.join(args.data_path,'EXTEND_image_quickdraw')
        elif type_skim == 'sketch':
            self.dir_file = os.path.join(args.data_path, 'QuickDraw_sketches')
        else:
            NameError(type_skim + ' not implemented!')

        self.fnames, self.cls = get_file_list(self.dir_file, self.set_class, type_skim)
        self.loader = default_image_loader

    def __getitem__(self, index):
        label = self.cls[index]
        fname = os.path.join(self.dir_file, label, self.fnames[index])
        photo = self.transform(self.loader(fname))
        lbl = self.dicts_class.get(label)

        return photo, fname, lbl

    def __len__(self):
        return len(self.fnames)

    def get_classDict(self):
        return self.set_class


class QuickDraw_Extended_train(data.Dataset):
    def __init__(self, args, train_class, dicts_class, class_emb, vocab, transform=None):
        self.transform = transform
        self.train_class = train_class
        self.dicts_class = dicts_class
        self.word2vec = class_emb
        self.vocab = vocab



        self.dir_image = os.path.join(args.data_path, 'QuickDraw_images_final')
        self.dir_sketch = os.path.join(args.data_path, 'QuickDraw_sketches_final')
        self.loader = default_image_loader
        self.fnames_sketch, self.cls_sketch = get_file_list(self.dir_sketch,
                                                            self.train_class, 'sketch')
        self.temp = 0.1  # Similarity temperature
        self.w2v_sim = np.exp(-np.square(cdist(self.word2vec, self.word2vec, 'euclidean')) / self.temp)

    def __getitem__(self, index):
        # Read sketch

        fname = os.path.join(self.dir_sketch, self.cls_sketch[index], self.fnames_sketch[index])
        sketch = self.loader(fname)
        sketch = self.transform(sketch)

        # Target
        label = self.cls_sketch[index]
        lbl = self.dicts_class.get(label)

        # Word 2 Vec (Semantics)
        w2v = torch.FloatTensor(self.word2vec[self.vocab.index(label), :])

        # Negative class
        # Hard negative
        sim = self.w2v_sim[self.vocab.index(label), :]
        possible_classes = [x for x in self.train_class if x != label]
        sim = [sim[self.vocab.index(x)] for x in possible_classes]
        # Similarity to probability
        norm = np.linalg.norm(sim, ord=1)
        sim = sim / norm
        label_neg = np.random.choice(possible_classes, 1, p=sim)[0]
        lbl_neg = self.dicts_class.get(label_neg)

        # Positive image
        # The constraint according to the ECCV 2018
        # fname = os.path.join(self.dir_image, label, (fname.split('/')[-1].split('-')[0]+'.jpg'))
        fname = get_random_file_from_path(os.path.join(self.dir_image, label))
        image = self.transform(self.loader(fname))

        fname = get_random_file_from_path(os.path.join(self.dir_image, label_neg))
        image_neg = self.transform(self.loader(fname))

        return sketch, image, image_neg, w2v, lbl, lbl_neg

    def __len__(self):
        return len(self.fnames_sketch)

    def get_classDict(self):
        return self.train_class

