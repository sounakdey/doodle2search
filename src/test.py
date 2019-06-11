# Python modules
import torch
from torch.autograd.variable import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

import os
import numpy as np
import time
from sklearn.metrics import average_precision_score
import math
import multiprocessing
from joblib import Parallel, delayed
import pickle

# Own modules
from scipy.spatial.distance import cdist
from options import Options
from utils import load_checkpoint, rec, precak
from models.encoder import EncoderCNN
from data.generator_train import load_data


def test(im_loader, sk_loader, model, args, dict_class=None):
    # Start counting time
    end = time.time()

    # Switch to test mode
    im_net, sk_net = model
    im_net.eval()
    sk_net.eval()
    torch.set_grad_enabled(False)
    acc_fnames_im = []
    acc_fnames_sk = []

    for i, (im, fname, target) in enumerate(im_loader):
        # Data to Variable
        if args.cuda:
            im, target = im.cuda(), target.cuda()

        # Process
        out_feat, _ = im_net(im)
                

        # Filename of the images for qualitative
        acc_fnames_im.append(fname)

        if i == 0:
            acc_im_em = out_feat.cpu().data.numpy()
            acc_cls_im = target.cpu().data.numpy()

        else:
            acc_im_em = np.concatenate((acc_im_em, out_feat.cpu().data.numpy()), axis=0)
            acc_cls_im = np.concatenate((acc_cls_im, target.cpu().data.numpy()), axis=0)

    for i, (sk, fname, target) in enumerate(sk_loader):
        # Data to Variable
        if args.cuda:
            sk, target = sk.cuda(), target.cuda()

        # Process
        out_feat, _ = sk_net(sk)

        # Filename of the images for qualitative
        acc_fnames_sk.append(fname)

        if i == 0:
            acc_sk_em = out_feat.cpu().data.numpy()
            acc_cls_sk = target.cpu().data.numpy()
        else:
            acc_sk_em = np.concatenate((acc_sk_em, out_feat.cpu().data.numpy()), axis=0)
            acc_cls_sk = np.concatenate((acc_cls_sk, target.cpu().data.numpy()), axis=0)

    # Evaluation
    # Distance Measure
    distance = cdist(acc_sk_em, acc_im_em, 'euclidean') # L1 same as Manhattan, Cityblock
    # distance = cdist(acc_sk_em, acc_im_em, 'cosine')/2    # Distance between 0 and 2

    # Now the average precision score needs probability estimate of the class
    # 1. Though the values are from 0-1 the probabilty has reverse interpretation than the distance
    # 2. Hence Similarity
    # sim = 1 - distance
    sim = 1/(1+distance)

    # Save values
    str_sim = (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0)) * 1

    nq = str_sim.shape[0]
    num_cores = min(multiprocessing.cpu_count(), 32)
    # # -sim because values in similarity means 0= un-similar 1= very-similar
    # arg_sort_sim = (-sim).argsort()
    # sort_sim = []
    # sort_lst = []
    # for indx in range(0, arg_sort_sim.shape[0]):
    #     sort_sim.append(sim[indx, arg_sort_sim[indx, :]])
    #     sort_lst.append(str_sim[indx, arg_sort_sim[indx, :]])
    #
    # sort_sim = np.array(sort_sim)
    # sort_str_sim = np.array(sort_lst)


    # aps_200 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(sort_str_sim[iq, 0:200], sort_sim[iq, 0:200])\
    #                                      for iq in range(nq))
    # aps_200_actual = [0.0 if math.isnan(x) else x for x in aps_200]
    # map_200 = np.mean(aps_200_actual)
    #
    # # Precision@200 means at the place 200th
    # precision_200 = np.mean(sort_str_sim[:, 200])
    #



    # mpreck, reck = precak(sim, str_sim, k=5)


    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))

    # if dict_class is not None:
    #     dict_class = {v: k for k, v in dict_class.items()}
    #     diff_class = set(acc_cls_sk)
    #     for cls in diff_class:
    #         ind = acc_cls_sk == cls
    #         print('Recall {} class {}'.format(str(np.array(reck)[ind].mean()), dict_class[cls]))

    if dict_class is not None:
        dict_class = {v: k for k, v in dict_class.items()}
        diff_class = set(acc_cls_sk)
        for cls in diff_class:
            ind = acc_cls_sk == cls
            print('mAP {} class {}'.format(str(np.array(aps)[ind].mean()), dict_class[cls]))

    map_ = np.mean(aps)

    if args.plot:
        # Qualitative Results
        # acc_fnames_im = []
        # acc_fnames_sk = []
        flatten_acc_fnames_sk = [item for sublist in acc_fnames_sk for item in sublist]
        flatten_acc_fnames_im = [item for sublist in acc_fnames_im for item in sublist]

        retrieved_im_fnames = []
        # Just a try
        retrieved_im_true_false = []
        for i in range(0, sim.shape[0]):
            sorted_indx = np.argsort(sim[i, :])[::-1]
            retrieved_im_fnames.append(list(np.array(flatten_acc_fnames_im)[sorted_indx][:args.num_retrieval]))
            # Just a try
            retrieved_im_true_false.append(list(np.array(str_sim[i])[sorted_indx][:args.num_retrieval]))

        with open('../data/sketches.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([flatten_acc_fnames_sk], f)

        with open('../data/retrieved_im_fnames.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([retrieved_im_fnames], f)

        with open('../data/retrieved_im_true_false.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([retrieved_im_true_false], f)



    # Measure elapsed time
    batch_time = time.time() - end

    print('* mAP {mean_ap:.3f}; Avg Time x Batch {b_time:.3f}'.format(mean_ap=map_, b_time=batch_time))
    return map_ #, map_200, precision_200


def main():
    print('Prepare data')
    transform = transforms.Compose([transforms.ToTensor()])
    _, [valid_sk_data, valid_im_data], [test_sk_data, test_im_data], dict_class = load_data(args, transform)

    valid_sk_loader = DataLoader(valid_sk_data, batch_size=3 * args.batch_size, num_workers=args.prefetch,
                                 pin_memory=True)
    valid_im_loader = DataLoader(valid_im_data, batch_size=3 * args.batch_size, num_workers=args.prefetch,
                                 pin_memory=True)
    test_sk_loader = DataLoader(test_sk_data, batch_size=3 * args.batch_size, num_workers=args.prefetch,
                                pin_memory=True)
    test_im_loader = DataLoader(test_im_data, batch_size=3 * args.batch_size, num_workers=args.prefetch,
                                pin_memory=True)

    print('Create model')
    im_net = EncoderCNN(out_size=args.emb_size, attention=args.attn)
    sk_net = EncoderCNN(out_size=args.emb_size, attention=args.attn)

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        im_net = nn.DataParallel(im_net, device_ids=list(range(args.ngpu)))
        sk_net = nn.DataParallel(sk_net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        im_net, sk_net = im_net.cuda(), sk_net.cuda()

    print('Loading model')
    checkpoint = load_checkpoint(args.load)
    im_net.load_state_dict(checkpoint['im_state'])
    sk_net.load_state_dict(checkpoint['sk_state'])
    print('Loaded model at epoch {epoch} and mAP {mean_ap}%'.format(epoch=checkpoint['epoch'],
                                                                    mean_ap=checkpoint['best_map']))

    print('***Valid***')
    #map_valid = test(valid_im_loader, valid_sk_loader, [im_net, sk_net], args, dict_class)
    print('***Test***')
    map_test = test(test_im_loader, test_sk_loader, [im_net, sk_net], args, dict_class)


if __name__ == '__main__':
    # Parse options
    args = Options(test=True).parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Check Test and Load
    if args.load is None:
        raise Exception('Cannot test without loading a model.')

    main()
