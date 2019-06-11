# -*- coding: utf-8 -*- 
from __future__ import print_function, division

"""
Doodle to Search
"""

# Python modules
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

import glob
import numpy as np
import time
import os


# Own modules
from options import Options
from Logger import LogMetric
from utils import save_checkpoint, load_checkpoint
from test import test
from models.encoder import EncoderCNN
from data.generator_train import load_data
from loss.loss import DetangledJoinDomainLoss
import json


def adjust_learning_rate(optimizer, epoch):
    """
        Updates the learning rate given an schedule and a gamma parameter.
    """
    if epoch in args.schedule:
        args.learning_rate *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate


def train(data_loader, model, optimizer, cuda, criterion, epoch, log_int=20):
    batch_time = LogMetric.AverageMeter()
    losses_sem = LogMetric.AverageMeter()
    losses_dom = LogMetric.AverageMeter()
    losses_spa = LogMetric.AverageMeter()
    losses = LogMetric.AverageMeter()

    # switch to train mode
    im_net, sk_net = model
    im_net.train()
    sk_net.train()
    torch.set_grad_enabled(True)

    end = time.time()
    for i, (sk, im, im_neg, w2v, _, _) in enumerate(data_loader):
        # Prepare input data
        if cuda:
            im, im_neg, sk, w2v = im.cuda(), im_neg.cuda(), sk.cuda(), w2v.cuda()
        
        optimizer.zero_grad()
        bs = im.size(0)
        # Output
        ## Image
        im_feat, _ = im_net(im) # Image encoding and projection to semantic space

        ## Image Negative
        # Encode negative image
        im_feat_neg, _ = im_net(im_neg)  # Image encoding and projection to semantic space
            
        ## Sketch
        sk_feat, _ = sk_net(sk) # Sketch encoding and projection to semantic space
        
        # LOSS
        loss, loss_sem, loss_dom, loss_spa = criterion(im_feat, sk_feat, w2v, im_feat_neg, i)
        
        # Gradiensts and update
        loss.backward()
        optimizer.step()
                
        # Save values
        losses_sem.update(loss_sem.item(), bs)
        losses_dom.update(loss_dom.item(), bs)
        losses_spa.update(loss_spa.item(), bs)
        losses.update(loss.item(), bs)
        batch_time.update(time.time() - end, bs)
        end = time.time()

        if log_int > 0 and i%log_int == 0:
            print('Epoch: [{0}]({1}/{2}) Average Loss {loss.avg:.3f} ( Sem: {loss_sem.avg} + Dom: {loss_dom.avg} + Spa: {loss_spa.avg}); Avg Time x Batch {b_time.avg:.3f}'
                .format(epoch, i, len(data_loader), loss=losses, loss_sem=losses_sem, loss_dom=losses_dom, loss_spa=losses_spa, b_time=batch_time))
    print('Epoch: [{0}] Average Loss {loss.avg:.3f} ( {loss_sem.avg} + {loss_dom.avg} + {loss_spa.avg} ); Avg Time x Batch {b_time.avg:.3f}'
        .format(epoch, loss=losses, loss_sem=losses_sem, loss_dom=losses_dom, loss_spa=losses_spa, b_time=batch_time))
    return losses, losses_sem, losses_dom, losses_spa


def main():
    print('Prepare data')
    transform = transforms.Compose([transforms.ToTensor()])
    train_data, [valid_sk_data, valid_im_data], [test_sk_data, test_im_data], dict_class = load_data(args, transform)    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    valid_sk_loader = DataLoader(valid_sk_data, batch_size=3*args.batch_size, num_workers=args.prefetch, pin_memory=True)
    valid_im_loader = DataLoader(valid_im_data, batch_size=3*args.batch_size, num_workers=args.prefetch, pin_memory=True)
    test_sk_loader = DataLoader(test_sk_data, batch_size=3*args.batch_size, num_workers=args.prefetch, pin_memory=True)
    test_im_loader = DataLoader(test_im_data, batch_size=3*args.batch_size, num_workers=args.prefetch, pin_memory=True)

    if args.log:
        if args.dataset == 'quickdraw_extend':
            pass
        elif not args.attn:
            pass
        else:
            # Save some images to plot attention
            rand_samples_sk = np.random.randint(0, high=len(valid_sk_data), size=5)
            rand_samples_im = np.random.randint(0, high=len(valid_im_data), size=5)
            for i in range(len(rand_samples_sk)):
                sk, _, lbl_sk = valid_sk_data[rand_samples_sk[i]]
                im, _, lbl_im = valid_im_data[rand_samples_im[i]]
                if args.cuda:
                    sk, im = sk.cuda(), im.cuda()
                if i == 0:
                    sk_log = sk.unsqueeze(0)
                    im_log = im.unsqueeze(0)
                    sk_lbl_log = [lbl_sk]
                    im_lbl_log = [lbl_im]
                else:
                    sk_log = torch.cat((sk_log, sk.unsqueeze(0)), dim=0)
                    im_log = torch.cat((im_log, im.unsqueeze(0)), dim=0)
                    sk_lbl_log.append(lbl_sk)
                    im_lbl_log.append(lbl_im)
    
    print('Create trainable model')
    if args.nopretrain:
        print('\t* Loading a pretrained model')

    im_net = EncoderCNN(out_size=args.emb_size, pretrained=args.nopretrain, attention=args.attn)
    sk_net = EncoderCNN(out_size=args.emb_size, pretrained=args.nopretrain, attention=args.attn)

    print('Loss, Optimizer & Evaluation')
    criterion = DetangledJoinDomainLoss(emb_size=args.emb_size, w_sem=args.w_semantic, w_dom=args.w_domain, w_spa=args.w_triplet, lambd=args.grl_lambda)
    criterion.train()
    optimizer = torch.optim.SGD(list(im_net.parameters()) + list(sk_net.parameters()) + list(criterion.parameters()), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)

    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel')
        im_net = nn.DataParallel(im_net, device_ids=list(range(args.ngpu)))
        sk_net = nn.DataParallel(sk_net, device_ids=list(range(args.ngpu)))
        criterion = nn.DataParallel(criterion, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        im_net, sk_net = im_net.cuda(), sk_net.cuda()
        criterion = criterion.cuda()

    start_epoch = 0
    best_map = 0
    early_stop_counter = 0
    if args.load is not None:
        print('Loading model')
        checkpoint = load_checkpoint(args.load)
        im_net.load_state_dict(checkpoint['im_state'])
        sk_net.load_state_dict(checkpoint['sk_state'])
        criterion.load_state_dict(checkpoint['criterion'])
        start_epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        print('Loaded model at epoch {epoch} and mAP {mean_ap}%'.format(epoch=checkpoint['epoch'],mean_ap=checkpoint['best_map']))

    print('***Train***')
       
    for epoch in range(start_epoch, args.epochs):
        # Update learning rate
        adjust_learning_rate(optimizer, epoch)

        loss_train, loss_sem, loss_dom, loss_spa = train(train_loader, [im_net, sk_net], optimizer, args.cuda, criterion, epoch, args.log_interval)
        map_valid = test(valid_im_loader, valid_sk_loader, [im_net, sk_net], args)

        # Logger
        if args.log:
            im_net.eval()
            sk_net.eval()
            if args.dataset == 'quickdraw_extend':
                pass
            elif not args.attn:
                pass
            else:
                with torch.set_grad_enabled(False):
                    # Images
                    _, attn_im = im_net(im_log)
                    attn_im = nn.Upsample(size=(im_log[0].size(1), im_log[0].size(2)), mode='bilinear', align_corners=False)(attn_im)
                    attn_im = attn_im - attn_im.view((attn_im.size(0), -1)).min(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    attn_im =  1 - attn_im/attn_im.view((attn_im.size(0), -1)).max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    _, attn_sk = sk_net(sk_log)
                    attn_sk = nn.Upsample(size=(sk_log[0].size(1), sk_log[0].size(2)), mode='bilinear', align_corners=False)(attn_sk)
                    attn_sk = attn_sk - attn_sk.view((attn_sk.size(0), -1)).min(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    attn_sk =  attn_sk/attn_sk.view((attn_sk.size(0), -1)).max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    for i in range(im_log.size(0)):
                        plt_im = torch.cat([im_log[i], attn_im[i]], dim=0)
                        nam = list(dict_class.keys())[list(dict_class.values()).index(im_lbl_log[i])]
                        logger.add_image('im{}_{}'.format(i, nam), plt_im)
                        nam = list(dict_class.keys())[list(dict_class.values()).index(sk_lbl_log[i])]
                        plt_im = sk_log[i]*attn_sk[i]
                        logger.add_image('sk{}_{}'.format(i, nam), plt_im)

            # Scalars
            logger.add_scalar('loss_train', loss_train.avg)
            logger.add_scalar('loss_sem', loss_sem.avg)
            logger.add_scalar('loss_dom', loss_dom.avg)
            logger.add_scalar('loss_spa', loss_spa.avg)
            logger.add_scalar('map_valid', map_valid)
            logger.add_scalar('learning_rate', args.learning_rate)
            logger.step()
        
        # Early-Stop
        if map_valid > best_map:
            best_map = map_valid
            best_epoch = epoch + 1
            early_stop_counter = 0
            if args.save is not None:
                save_checkpoint({'epoch': epoch + 1, 'im_state': im_net.state_dict(), 'sk_state': sk_net.state_dict(), 'criterion': criterion.state_dict(), 'best_map': best_map}, directory=args.save, file_name='checkpoint')
        else:
            if early_stop_counter == args.early_stop:
                break
            early_stop_counter += 1

    # Load Best model in case of save it
    if args.save is not None:
        print('Loading best  model')
        best_model_file = os.path.join(args.save, 'checkpoint.pth')
        checkpoint = load_checkpoint(best_model_file)
        im_net.load_state_dict(checkpoint['im_state'])
        sk_net.load_state_dict(checkpoint['sk_state'])
        best_map = checkpoint['best_map']
        best_epoch = checkpoint['epoch']
        print('Best model at epoch {epoch} and mAP {mean_ap}%'.format(epoch=checkpoint['epoch'],mean_ap=checkpoint['best_map']))

    print('***Test***')

    map_test= test(test_im_loader, test_sk_loader, [im_net, sk_net], args)
    print('Test mAP {mean_ap}%'.format(mean_ap=map_test))
    # print('Test mAP@200 {map_200}%'.format(map_200=map_200))
    # print('Test Precision@200 {prec_200}%'.format(prec_200=precision_200))

    if args.exp_idf is not None:
        with open(os.path.join(args.log, 'results.txt'), 'w') as fp:
            print('Epoch: {best_epoch:.3f}'.format(best_epoch=best_epoch), file=fp)
            print('Valid: {mean_ap:.3f}'.format(mean_ap=best_map), file=fp)
            print('Test mAP: {mean_ap:.3f}'.format(mean_ap=map_test), file=fp)
            # print('Test mAP@200: {map_200:.3f}'.format(map_200=map_200), file=fp)
            # print('Test Precision@200: {prec_200:.3f}'.format(prec_200=precision_200), file=fp)



if __name__ == '__main__':
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    
    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.exp_idf is not None:
        if not os.path.isdir(os.path.join('./checkpoint', args.exp_idf)):
            os.makedirs(os.path.join('./checkpoint', args.exp_idf))
        args.save = os.path.join('./checkpoint', args.exp_idf)
        args.log = os.path.join('./checkpoint', args.exp_idf)+'/'


    if args.log is not None:
        print('Initialize logger')
        log_dir = args.log + '{}_run-batchSize_{}/' \
                .format(len(glob.glob(args.log + '*_run-batchSize_{}'.format(args.batch_size))), args.batch_size)

        args.save = log_dir
        # Create logger
        print('Log dir:\t' + log_dir)
        logger = LogMetric.Logger(log_dir, force=True)
        with open(os.path.join(args.save, 'params.txt'), 'w') as fp:
            for key, val in vars(args).items():
                fp.write('{} {}\n'.format(key, val))

    main()

