#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'.')

# Datasets
from data.sketchy_extended import Sketchy_Extended
# from data.sbir_sketchy_extended import SBIR_Sketchy_Extended
from data.tuberlin_extended import TUBerlin_Extended
from data.quickdraw_extended import QuickDraw_Extended
# from data.quickdraw_extended_256 import QuickDraw_Extended_256


def load_data(args, transform = None):

    if args.dataset == 'sketchy_extend':
        return Sketchy_Extended(args, transform)
    elif args.dataset == 'quickdraw_extend':
        return QuickDraw_Extended(args, transform)
    # elif args.dataset == 'quickdraw_extend_256':
    #     return QuickDraw_Extended_256(args, transform)
    # elif args.dataset == 'sbir_sketchy_extend':
    #     return SBIR_Sketchy_Extended(args, transform)
    elif args.dataset == 'tuberlin_extend':
        return TUBerlin_Extended(args, transform)
    else:
        sys.exit()

    raise NameError(args.dataset + ' not implemented!')


if __name__=='__main__':
    from options import Options
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import numpy as np

    dict_by_value = lambda dic, value: list(dic.keys())[list(dic.values()).index(value)]
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader, [valid_sk_loader, valid_im_loader], [test_sk_loader, test_im_loader], dict_class = load_data(args, transform)

    print('\n--- Train Data ---')
    print('\t* Length: {}'.format(len(train_loader)))
    print('\t* Classes: {}'.format(train_loader.get_classDict()))
    print('\t* Num Classes. {}'.format(len(train_loader.get_classDict())))
    # train_lbl = np.sort(np.unique([[lbl, lbl_neg] for _, _, _, _, lbl, lbl_neg in train_loader]))
    # train_dict = np.sort([dict_class[i] for i in train_loader.get_classDict()])
    # if np.all(train_lbl==train_dict):
    #     print('\t* Same Classes: True')
    # else:
    #     print('\t* Same Classes: False')
    
    num_samples = 7
    rand_samples = np.random.randint(0, high=len(train_loader), size=num_samples)
    f, axarr = plt.subplots(3,num_samples)
    for i in range(len(rand_samples)):
        sk, im, im_neg, w2v, lbl, lbl_neg = train_loader[rand_samples[i]]
        axarr[0,i].imshow(sk.permute(1,2,0).numpy())
        axarr[0,i].set_title(dict_by_value(dict_class, lbl))
        axarr[0,i].axis('off')
        
        axarr[1,i].imshow(im.permute(1,2,0).numpy())
        axarr[1,i].axis('off')
        
        axarr[2,i].imshow(im_neg.permute(1,2,0).numpy())
        axarr[2,i].set_title(dict_by_value(dict_class, lbl_neg))
        axarr[2,i].axis('off')
    plt.show()

    print('\n--- Valid Data ---')
    print('\t* Length Sketch: {}'.format(len(valid_sk_loader)))
    print('\t* Length Image: {}'.format(len(valid_im_loader)))
    # sk_valid_lbl = np.sort(np.unique([lbl for _, _, lbl in valid_sk_loader]))
    # sk_valid_dict = np.sort([dict_class[i] for i in valid_sk_loader.get_classDict()])
    # im_valid_lbl = np.sort(np.unique([lbl for _, _, lbl in valid_im_loader]))
    # im_valid_dict = np.sort([dict_class[i] for i in valid_im_loader.get_classDict()])
    # if np.all(sk_valid_lbl==sk_valid_dict)and np.all(im_valid_lbl==im_valid_dict) and np.all(sk_valid_dict==im_valid_lbl) and valid_sk_loader.get_classDict()==valid_im_loader.get_classDict():
    #     print('\t* Same Classes: True')
    # else:
    #     print('\t* Same Classes: False')
    print('\t* Classes: {}'.format(valid_sk_loader.get_classDict()))
    print('\t* Num Classes. {}'.format(len(valid_sk_loader.get_classDict())))
    
    rand_samples_sk = np.random.randint(0, high=len(valid_sk_loader), size=num_samples)
    rand_samples_im = np.random.randint(0, high=len(valid_im_loader), size=num_samples)
    f, axarr = plt.subplots(2,num_samples)
    for i in range(len(rand_samples_sk)):
        sk, fname, lbl = valid_sk_loader[rand_samples_sk[i]]
        axarr[0,i].imshow(sk.permute(1,2,0).numpy())
        axarr[0,i].set_title(dict_by_value(dict_class, lbl))
        axarr[0,i].axis('off')
        
        im, fname, lbl = valid_im_loader[rand_samples_im[i]]
        axarr[1,i].imshow(im.permute(1,2,0).numpy())
        axarr[1,i].set_title(dict_by_value(dict_class, lbl))
        axarr[1,i].axis('off')
    plt.show()


    print('\n--- Test Data ---')
    print('\t* Length Sketch: {}'.format(len(test_sk_loader)))
    print('\t* Length mage: {}'.format(len(test_im_loader)))
    # sk_test_lbl = np.sort(np.unique([lbl for _, _, lbl in test_sk_loader]))
    # sk_test_dict = np.sort([dict_class[i] for i in test_sk_loader.get_classDict()])
    # im_test_lbl = np.sort(np.unique([lbl for _, _, lbl in test_im_loader]))
    # im_test_dict = np.sort([dict_class[i] for i in test_im_loader.get_classDict()])
    # if np.all(sk_test_lbl==sk_test_dict)and np.all(im_test_lbl==im_test_dict) and np.all(sk_test_dict==im_test_lbl) and test_sk_loader.get_classDict()==test_im_loader.get_classDict():
    #     print('\t* Same Classes: True')
    # else:
    #     print('\t* Same Classes: False')
    print('\t* Classes: {}'.format(test_sk_loader.get_classDict()))
    print('\t* Num Classes. {}'.format(len(test_sk_loader.get_classDict())))

    rand_samples_sk = np.random.randint(0, high=len(test_sk_loader), size=num_samples)
    rand_samples_im = np.random.randint(0, high=len(test_im_loader), size=num_samples)
    f, axarr = plt.subplots(2,num_samples)
    for i in range(len(rand_samples_sk)):
        sk, fname, lbl = test_sk_loader[rand_samples_sk[i]]
        axarr[0,i].imshow(sk.permute(1,2,0).numpy())
        axarr[0,i].set_title(dict_by_value(dict_class, lbl))
        axarr[0,i].axis('off')
        
        im, fname, lbl = test_im_loader[rand_samples_im[i]]
        axarr[1,i].imshow(im.permute(1,2,0).numpy())
        axarr[1,i].set_title(dict_by_value(dict_class, lbl))
        axarr[1,i].axis('off')
    plt.show()


    print('\n--- Disjoin ---')
    # if np.intersect1d(train_lbl, sk_valid_lbl).size == 0 and np.intersect1d(train_lbl, sk_test_lbl).size == 0 and np.intersect1d(sk_test_lbl, sk_valid_lbl).size == 0:
    #     print('\t* Disjoin: True')
    # else:
    #     print('\t* Disjoin: False')

