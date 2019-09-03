import matplotlib
matplotlib.use('Agg')
from data import *
from layers.box_utils import *
from utils.augmentations import SSDAugmentation
from ssd import build_ssd
import os, datetime
import sys
sys.path.append(os.path.expanduser("~/Documents"))
import omni_torch.visualize.basic as vb
from omni_torch.networks.optimizer import *
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from args import prepare_args
import mmdet.ops.dcn as dcn
from layers.visualization import *


args = prepare_args(VOC_ROOT)
TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def avg(list):
    return sum(list) / len(list)

def fit(args, cfg, net, dataset, optimizer, is_train=True):
    if is_train:
        net.train()
        Loss_L, Loss_C = [], []
    else:
        net.eval()
        eval_results = []
    start_time = time.time()
    for batch_idx, (images, targets, img_shape) in enumerate(dataset):
        if not is_train and batch_idx >= 10:
            break
        # if not net.fix_size:
        # assert images.size(0) == 1, "batch size for dynamic input shape can only be 1 for 1 GPU RIGHT NOW!"
        if len(targets) == 0:
            continue
        images = images.cuda()
        targets = [ann.cuda() for ann in targets]
        targets_idx_ = torch.cuda.LongTensor([ann.size(0) for ann in targets])
        targets_idx = torch.cuda.LongTensor([sum(targets_idx_[:_idx]) for _idx in range(len(targets_idx_))])
        if batch_idx == 0 and args.visualize_box:
            #visualize_bbox(args, cfg, images, targets, net.module.priors[0], batch_idx)
            pass
        y_idx = torch.stack([targets_idx, targets_idx_], dim=1)
        y = torch.cat(targets, dim=0).repeat(torch.cuda.device_count(), 1)
        out1, out2 = net(images, y, y_idx, deform_map=False, test=(not is_train))
        if is_train:
            # During train phase, out1 represent loss_l and out2 represent loss_c
            loss = out1.mean() + out2.mean()
            Loss_L.append(float(out1.mean().data))
            Loss_C.append(float(out2.mean().data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #args.curr_epoch += 1
        else:
            # Turn the input param detector into None so as to
            # Experiment with Detector's Hyper-parameters
            detections, reg_boxes = out1
            eval_result = evaluate(images, detections.data, targets, batch_idx, 0.1,
                                   visualize=False, post_combine=True)
            eval_results.append(eval_result)
    if is_train:
        print(" --- loc loss: %.4f, conf loss: %.4f, at epoch %04d, cost %.2f seconds ---" %
              (avg(Loss_L), avg(Loss_C), args.curr_epoch + 1, time.time() - start_time))
        args.curr_epoch += 1
        return avg(Loss_L), avg(Loss_C)
    else:
        eval_results = list(map(list, zip(*eval_results)))
        print(" --- accuracy=%.4f, precision=%.4f, recall=%.4f, f1-score=%.4f, cost %.2f seconds ---" %
          (avg(eval_results[0]), avg(eval_results[1]), avg(eval_results[2]),
           avg(eval_results[3]), time.time() - start_time))
        # represent accuracy, precision, recall, f1_score
        return avg(eval_results[0]), avg(eval_results[1]), avg(eval_results[2]), avg(eval_results[3])

def val(args, cfg, net, dataset, optimizer):
    with torch.no_grad():
        return fit(args, cfg, net, dataset, optimizer, is_train=False)

def evaluate(img, detections, targets, batch_idx, threshold, visualize=False, post_combine=False):
    eval_result = {}
    #save_dir = os.path.expanduser("~/Pictures/")
    w = img.size(3)
    h = img.size(2)
    accu, pre, rec, f1 = [], [], [], []
    for i in range(detections.size(0)):
        gt_cls = targets[i][:, -1].data
        tar = targets[i][:, :-1].data
        gt_boxes = [[] for _ in range(20)]
        for _i, cls in enumerate(gt_cls):
            gt_boxes[int(cls)].append(tar[_i])

        # enumerate through all classes
        for j in range(1, detections.size(1)):
            idx = detections[i, j, :, 0] >= threshold
            boxes = detections[i, j, idx, 1:]
            if boxes.size(0) == 0 and len(gt_boxes[j-1]) == 0:
                continue
            elif boxes.size(0) != 0 and len(gt_boxes[j-1]) == 0:
                accuracy, precision, recall, f1_score = 0, 0, 1, 0
            elif boxes.size(0) == 0 and len(gt_boxes[j-1]) != 0:
                accuracy, precision, recall, f1_score = 0, 1, 0, 0
            else:
                gt = torch.stack(gt_boxes[j - 1], 0)
                jac = jaccard(boxes, gt)
                overlap, idx = jac.max(1, keepdim=True)
                # This is not DetEval
                positive_pred = boxes[overlap.squeeze(1) > 0.5]
                negative_pred = boxes[overlap.squeeze(1) <= 0.5]
                if negative_pred.size(0) == 0:
                    negative_pred = tuple()
                #print_box(blue_boxes=positive_pred, green_boxes=gt_boxes, red_boxes=negative_pred,
                          #img=vb.plot_tensor(args, img, margin=0), save_dir=save_dir)

                accuracy, precision, recall = measure(positive_pred, gt, width=w, height=h)
                if (recall + precision) < 1e-3:
                    f1_score = 0
                else:
                    f1_score = 2 * (recall * precision) / (recall + precision)
                if visualize and threshold == 0.1 and i == 0:
                    pred = [[float(coor) for coor in area] for area in positive_pred]
                    gt = [[float(coor) for coor in area] for area in gt_boxes]
                    #print_box(negative_pred, green_boxes=gt, blue_boxes=pred, idx=batch_idx,
                              #img=vb.plot_tensor(args, img, margin=0), save_dir=args.val_log)
            accu.append(accuracy)
            pre.append(precision)
            rec.append(recall)
            f1.append(f1_score)
        #eval_result.update({threshold: [avg(accu), avg(pre), avg(rec), avg(f1)]})
    return avg(accu), avg(pre), avg(rec), avg(f1)


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    else:
        if type(m) is nn.ModuleList:
            for _m in m:
                if type(_m) == torch.nn.Linear:
                    torch.nn.init.xavier_normal_(_m.weight)
                elif type(_m) == torch.nn.Conv2d:
                    torch.nn.init.kaiming_normal_(_m.weight)
                elif isinstance(_m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(_m.weight, 1)
                    torch.nn.init.constant_(_m.bias, 0)
        elif type(m) is dcn.DeformConv:
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            pass

def main():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        train_set = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'], MEANS))
        val_set = None
    elif args.dataset == 'VOC':
        #if args.dataset_root == COCO_ROOT:
            #parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        train_set = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))
        train_set = data.DataLoader(train_set, args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=False, collate_fn=detection_collate,
                                      pin_memory=True)
        val_set = None
        """
        val_set = VOCDetection(args.voc_root, [('2007', "test")],
                               BaseTransform(args.img_size, (104, 117, 123)),
                               VOCAnnotationTransform())
        val_set = data.DataLoader(val_set, args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False, collate_fn=detection_collate,
                                    pin_memory=True)"""
    else:
        train_set, val_set = None, None

    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    ssd_net = build_ssd(args, 'train', cfg['min_dim'], cfg['num_classes'])

    args.curr_epoch = args.start_iter + args.ft_iter
    if args.resume:
        model_name = "%s_%s_%s.pth" % (args.basenet, args.img_size, args.ft_iter)
        print('Resuming training from %s...' % (model_name))
        weights = torch.load(os.path.join(args.save_folder, model_name))
        ssd_net.load_state_dict(weights)
    else:
        vgg_weights = torch.load(os.path.join(args.save_folder, args.basenet))
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
        print('Initializing weights...')
        ssd_net.extras.apply(weights_init)
        if args.implementation in ["header", "190709"]:
            ssd_net.header.apply(weights_init)
        elif args.implementation == "vanilla":
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net).cuda()
        cudnn.benchmark = True

    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "super":
        optimizer = SuperConv(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #criterion = MultiBoxLoss(cfg['num_classes'], args.overlap_threshold, True, 0,
                             #True, 3, 0.5, False, args.cuda, rematch=args.rematch)
    else:
        raise NotImplementedError()

    loc_loss, conf_loss = [], []
    accuracy, precision, recall, f1_score = [], [], [], []
    for epoch in range(args.max_iter):
        loc_avg, conf_avg = fit(args, cfg, net, train_set, optimizer, is_train=True)
        loc_loss.append(loc_avg)
        conf_loss.append(conf_avg)
        train_losses = [np.asarray(loc_loss), np.asarray(conf_loss)]
        if val_set is not None:
            #fit(args, cfg, net, val_set, optimizer, is_train=False)
            accu, pre, rec, f1 = val(args, cfg, net, val_set, optimizer)
            accuracy.append(accu)
            precision.append(pre)
            recall.append(rec)
            f1_score.append(f1)
            val_losses = [np.asarray(accuracy), np.asarray(precision),
                          np.asarray(recall), np.asarray(f1_score)]
        if epoch > 30 and epoch % 10 == 0:
            save_epoch = epoch + args.start_iter + args.ft_iter
            torch.save(net.module.state_dict(),
                       os.path.join(args.save_folder, '%s_%s_%s.pth' %
                                    (args.name, args.img_size, save_epoch)))
        if epoch >= 5:
            if val_set is None:
                vb.plot_curves(train_losses, ["location", "confidence"], save_path=args.val_log,
                               name=dt + "_" + args.name, window=5, fig_size=(18, 6),
                               bound={"low": 0.0, "high": 3.0}, title="Train Loss")
            else:
                vb.plot_multi_loss_distribution(
                    multi_line_data=[train_losses, val_losses],
                    multi_line_labels=[["location", "confidence"], ["Accuracy", "Precision", "Recall", "F1-Score"]],
                    save_path=args.val_log, window=5, name=dt + "_" + args.name, fig_size=(18, 12),
                    bound=[{"low": 0.0, "high": 3.0}, {"low": 0.0, "high": 1.0}],
                    titles=["Train Loss", "Validation Score"]
                )
        # 由于centroid可以向两个方向形成distortion，所以每个epoch后都需要重新创建一次
        # 以保证两个方向都能够受到distortion
        #net.module.create_centroid()

if __name__ == '__main__':
    main()