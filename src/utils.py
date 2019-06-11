#-*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Pytorch useful tools.
"""

import torch
import os
import errno
import numpy as np

import multiprocessing
from joblib import Parallel, delayed


def save_checkpoint(state, directory, file_name):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        print("=> loaded model '{}' (epoch {}, map {})".format(model_file, checkpoint['epoch'], checkpoint['best_map']))
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)

def rec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / len(act_set)
    return re

def prec(actual, predicted, k):
    act_set = set(actual)
    if k is not None:
        pred_set = set(predicted[:k])
    else:
        pred_set = set(predicted)
    pr = len(act_set & pred_set) / min(len(act_set), len(pred_set))
    return pr

def precak(sim, str_sim, k=None):
    act_lists = [np.nonzero(s)[0] for s in str_sim]
    pred_lists = np.argsort(-sim, axis=1)
    num_cores = min(multiprocessing.cpu_count(), 8)
    nq = len(act_lists)
    preck = Parallel(n_jobs=num_cores)(delayed(prec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    reck = Parallel(n_jobs=num_cores)(delayed(rec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    return np.mean(preck), reck

