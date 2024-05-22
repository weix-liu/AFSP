# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader_augment import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp,EFocalLoss

from model.utils.parser_func import parse_args, set_dataset_args
import itertools

from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv

import numpy

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name_target)
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))

    # -- Note: Use validation set and disable the flipped to enable faster loading.
    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # initilize the tensor holder here.
    im_data_s = torch.FloatTensor(1)
    im_data_s2 = torch.FloatTensor(1)
    im_data_w = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data_s = im_data_s.cuda()
        im_data_s2 = im_data_s2.cuda()
        im_data_w = im_data_w.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data_s = Variable(im_data_s)
    im_data_s2 = Variable(im_data_s2)
    im_data_w = Variable(im_data_w)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.SF_faster_rcnn.resnet_mt_afsp import resnet
    from model.SF_faster_rcnn.vgg16_mt_afsp import vgg16

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN_T = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN_T = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN_T = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()


    fasterRCNN.create_architecture()
    fasterRCNN_T.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr


    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()
        fasterRCNN_T.cuda()

    checkpoint = torch.load(args.load_name)
    fasterRCNN.load_state_dict(checkpoint['model'],strict=False)
    fasterRCNN_T.load_state_dict(checkpoint['model'],strict=False)

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (args.load_name))
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        fasterRCNN_T.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
        fasterRCNN_T = nn.DataParallel(fasterRCNN_T)
    iters_per_epoch = int(train_size / args.batch_size)

    count_iter = 0
    domain_data = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):

        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name_target)
        sampler_batch = sampler(train_size, args.batch_size)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,sampler=sampler_batch, num_workers=args.num_workers)

        # setting to train mode
        fasterRCNN.train()
        fasterRCNN_T.eval()

        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            optimizer.zero_grad()
            try:
                data = next(data_iter)
                data1 = next(data_iter)
            except:
                data_iter = iter(dataloader)
                data = next(data_iter)
                data1 = next(data_iter)

            eta = 1.0
            count_iter += 1

            gt_boxes_target = []

            # mixup teacher 1 ---------------------------------------------------------------------------------------------------
            data_w = data[0][:, 0, :, :, :]  # 不数据增强
            im_data_w.resize_(data_w.size()).copy_(data_w)
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).zero_()
            num_boxes.data.resize_(data[3].size()).zero_()

            with torch.no_grad():
                (
                    rois1,
                    cls_prob1,
                    bbox_pred,
                    _,
                    _,
                    _,
                    _,
                    _,
                    base_feat_T, base_feat1T, base_feat2T
                ) = fasterRCNN_T(im_data_w, im_info, gt_boxes, num_boxes)

            scores = cls_prob1.data
            boxes = rois1.data[:, :, 1:5]

            pooled_feat_base1 = fasterRCNN_T.RCNN_roi_align(base_feat_T, rois1.view(-1, 5)).mean(3).mean(2)
            c_prototype = fasterRCNN_T.global_pro
            cindex = torch.argmax(cls_prob1[0], 1)
            for c in range(fasterRCNN.n_classes):
                fg = pooled_feat_base1[cindex == c]
                if len(fg) > 0:
                    c_prototype[c,] = torch.mean(fg, 0)

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS
                        ).cuda()
                                + torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_MEANS
                        ).cuda()
                        )
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS
                        ).cuda()
                                + torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_MEANS
                        ).cuda()
                        )
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            pre_thresh = 0.0
            thresh = 0.7
            empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

            for j in range(1, len(imdb.classes)):
                inds = torch.nonzero(scores[:, j] > pre_thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    cls_dets_numpy = cls_dets.cpu().numpy()
                    # for i in range(np.minimum(10, cls_dets_numpy.shape[0])):
                    for i in range(int(cls_dets_numpy.shape[0])):
                        bbox = tuple(
                            int(np.round(x)) for x in cls_dets_numpy[i, :4]
                        )
                        score = cls_dets_numpy[i, -1]
                        if score > thresh:
                            gt_boxes_target.append(list(bbox[0:4]) + [j])
            mixup1_gt_boxes_target = gt_boxes_target.copy()

            # mixup teacher 2-------------------------------------------------------------------------------------------------------------------
            data_w = data1[0][:, 0, :, :, :]  # 不数据增强
            im_data_w.resize_(data_w.size()).copy_(data_w)
            im_info.data.resize_(data1[1].size()).copy_(data1[1])
            gt_boxes.data.resize_(data1[2].size()).zero_()
            num_boxes.data.resize_(data1[3].size()).zero_()

            with torch.no_grad():
                (
                    rois1,
                    cls_prob1,
                    bbox_pred,
                    _,
                    _,
                    _,
                    _,
                    _,
                    base_feat_T, base_feat1T, base_feat2T
                ) = fasterRCNN_T(im_data_w, im_info, gt_boxes, num_boxes)

            pooled_feat_base1 = fasterRCNN_T.RCNN_roi_align(base_feat_T, rois1.view(-1, 5)).mean(3).mean(2)
            cindex = torch.argmax(cls_prob1[0], 1)
            for c in range(fasterRCNN.n_classes):
                fg = pooled_feat_base1[cindex == c]
                if len(fg) > 0:
                    c_prototype[c,] = 0.5*c_prototype[c,] + 0.5*torch.mean(fg, 0)
            prototype_T = 0.7 * fasterRCNN_T.global_pro + c_prototype * 0.3


            scores = cls_prob1.data
            boxes = rois1.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS
                        ).cuda()
                                + torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_MEANS
                        ).cuda()
                        )
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS
                        ).cuda()
                                + torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_MEANS
                        ).cuda()
                        )
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            scores = scores.squeeze()

            pred_boxes = pred_boxes.squeeze()
            pre_thresh = 0.0
            thresh = 0.7
            empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

            for j in range(1, len(imdb.classes)):
                inds = torch.nonzero(scores[:, j] > pre_thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    cls_dets_numpy = cls_dets.cpu().numpy()
                    # for i in range(np.minimum(10, cls_dets_numpy.shape[0])):
                    for i in range(int(cls_dets_numpy.shape[0])):
                        bbox = tuple(
                            int(np.round(x)) for x in cls_dets_numpy[i, :4]
                        )
                        score = cls_dets_numpy[i, -1]
                        if score > thresh:
                            gt_boxes_target.append(list(bbox[0:4]) + [j])

            # mixup student --------------------------------------------------------------------------------------------------------------------------------
            gt_boxes_padding = torch.FloatTensor(cfg.MAX_NUM_GT_BOXES, 5).zero_()
            if len(gt_boxes_target) != 0:
                gt_boxes_numpy = torch.FloatTensor(gt_boxes_target)
                num_boxes_cpu = torch.LongTensor(
                    [min(gt_boxes_numpy.size(0), cfg.MAX_NUM_GT_BOXES)]
                )
                gt_boxes_padding[:num_boxes_cpu, :] = gt_boxes_numpy[:num_boxes_cpu]
            else:
                num_boxes_cpu = torch.LongTensor([0])

            width = max(data1[0][:, 1, :, :, :].size()[3],data[0][:, 1, :, :, :].size()[3])
            height = max(data1[0][:, 1, :, :, :].size()[2], data[0][:, 1, :, :, :].size()[2])
            mix_raw = torch.zeros(1,3,height,width)
            mix_raw[:,:,:data1[0][:, 1, :, :, :].size()[2],:data1[0][:, 1, :, :, :].size()[3]] = data1[0][:, 1, :, :, :]*0.5
            mix_raw[:, :, :data[0][:, 1, :, :, :].size()[2], :data[0][:, 1, :, :, :].size()[3]] = 0.5*data[0][:, 1, :, :, :]+0.5*mix_raw[:, :, :data[0][:, 1, :, :, :].size()[2], :data[0][:, 1, :, :, :].size()[3]]
            im_infodata = torch.tensor([[height,width,1.0]]).cuda()
            im_data_s.resize_(mix_raw.size()).copy_(mix_raw)
            im_info.data.resize_(im_infodata.size()).copy_(im_infodata)
            gt_boxes_padding = torch.unsqueeze(gt_boxes_padding, 0)
            gt_boxes.data.resize_(gt_boxes_padding.size()).copy_(gt_boxes_padding)
            num_boxes.data.resize_(num_boxes_cpu.size()).copy_(num_boxes_cpu)

            (
                rois,
                cls_prob,
                bbox_pred,
                rpn_loss_cls_s_fake,
                rpn_loss_box_s_fake,
                RCNN_loss_cls_s_fake,
                RCNN_loss_bbox_s_fake,
                rois_label_s_fake,
                base_feat_S, base_feat1S, base_feat2S
            ) = fasterRCNN(im_data_s, im_info, gt_boxes, num_boxes,using_adv=False)

            loss1 = rpn_loss_cls_s_fake.mean() + rpn_loss_box_s_fake.mean() + RCNN_loss_cls_s_fake.mean() + RCNN_loss_bbox_s_fake.mean()
            pooled_feat_base1 = fasterRCNN.RCNN_roi_align(fasterRCNN.transTS(base_feat_S), rois.view(-1, 5)).mean(3).mean(2)
            c_prototype = fasterRCNN.global_pro
            cindex = torch.argmax(cls_prob[0], 1)
            for c in range(fasterRCNN.n_classes):
                fg = pooled_feat_base1[cindex == c]
                if len(fg) > 0:
                    c_prototype[c,] = torch.mean(fg, 0)
            prototype = 0.7 * fasterRCNN.global_pro + c_prototype * 0.3


            # afsp---------------------------------------------------------------------------------------------------------------------------------------------
            data_s = data[0][:, 1, :, :, :]# 数据增强
            im_data_s2.resize_(data_s.size()).copy_(data_s)  # (1,3,600,1200)
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes_target = mixup1_gt_boxes_target

            gt_boxes_padding = torch.FloatTensor(cfg.MAX_NUM_GT_BOXES, 5).zero_()
            if len(gt_boxes_target) != 0:
                gt_boxes_numpy = torch.FloatTensor(gt_boxes_target)
                num_boxes_cpu = torch.LongTensor(
                    [min(gt_boxes_numpy.size(0), cfg.MAX_NUM_GT_BOXES)]
                )
                gt_boxes_padding[:num_boxes_cpu, :] = gt_boxes_numpy[:num_boxes_cpu]
            else:
                num_boxes_cpu = torch.LongTensor([0])

            using_adv = True


            gt_boxes_padding = torch.unsqueeze(gt_boxes_padding, 0)
            gt_boxes.data.resize_(gt_boxes_padding.size()).copy_(gt_boxes_padding)
            num_boxes.data.resize_(num_boxes_cpu.size()).copy_(num_boxes_cpu)
            (
                rois,
                cls_prob,
                bbox_pred,
                rpn_loss_cls_s_fake,
                rpn_loss_box_s_fake,
                RCNN_loss_cls_s_fake,
                RCNN_loss_bbox_s_fake,
                rois_label_s_fake,
                base_feat_S, base_feat1S, base_feat2S
            ) = fasterRCNN(im_data_s2, im_info, gt_boxes, num_boxes,using_adv=using_adv,alpha=0.5)

            loss2 = rpn_loss_cls_s_fake.mean() + rpn_loss_box_s_fake.mean() + RCNN_loss_cls_s_fake.mean() + RCNN_loss_bbox_s_fake.mean()

            pro_loss = torch.mean((prototype - prototype_T) ** 2)
            loss = loss1 + loss2 + 0.5 * pro_loss
            fasterRCNN.global_pro = prototype.detach()
            fasterRCNN_T.global_pro = prototype_T

            loss_temp += loss.item()
            loss.backward()

            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()


            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, pro_loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, pro_loss.item(), lr))
                loss_temp = 0
                start = time.time()

        alpha = 0.9
        for param_s, param_t,name in zip(fasterRCNN.parameters(), fasterRCNN_T.parameters(),fasterRCNN.state_dict()):
            param_t.data = alpha * param_t.data.detach() + (1 - alpha) * param_s.data.detach()

        save_name = os.path.join(output_dir,
                                 'SF_mt_afsp_target_{}_{}_{}_{}.pth'.format(args.dataset_t, args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN_T.module.state_dict() if args.mGPUs else fasterRCNN_T.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))




