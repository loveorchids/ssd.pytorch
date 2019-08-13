"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
from data import voc, coco
import imageio

from ssd import build_ssd
from layers import *
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
from args import prepare_args
from layers.box_utils import *

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

args = prepare_args(VOC_ROOT)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = str(args.year)
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = args.set
print("")
print("Evaluation on %s set: %s."%(set_type, devkit_path))


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=True):
    """
    rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname, [ovthresh], [use_07_metric])
        Top level function that does the PASCAL VOC evaluation.
    :param detpath: Path to detections, should produce the detection results file.
    :param annopath: Path to annotations, should be the xml annotations file.
    :param imagesetfile: Text file containing the list of images, one image per line.
    :param classname: Category name (duh)
    :param cachedir: irectory for caching the annotations
    :param ovthresh: Overlap threshold (default = 0.5)
    :param use_07_metric: Whether to use VOC07's 11 point AP computation(default True)
    :return:
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, '%s_annots.pkl'%set_type)
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):

    detector = Detect(num_classes, bkg_label=0, top_k=args.top_k,
                         conf_thresh=args.conf_threshold, nms_thresh=args.nms_threshold)

    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = os.path.join(os.getcwd(), "experiments", args.name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    det_file = os.path.join(output_dir, 'detections_%s.pkl'%args.iter)
    #progress = open(os.path.join(output_dir, "time_consumption_%s.txt"%args.iter), "w")

    start = time.time()
    for i in range(num_images):
        if i != 0 and i % 500 == 0:
            print("progress: %s/%s cost %.4f seconds."%(i, num_images, time.time() - start))
            start = time.time()
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        #_t['im_detect'].tic()
        out, deform_pyramid = net(x, deform_map=args.visualize_deformation, test=True)
        # reg_boxes and pred_box are in point form
        detections, reg_boxes = out
        detections = detections.data
        if len(deform_pyramid) > 0:
            cls = detections[0, 1:, :, 0]
            reg = detections[0, 1:, :, 1:]
            pred_box = reg[cls >= 0.1, :] * args.img_size
            visualize_deformation(voc, x, deform_pyramid, reg_boxes, net.priors[x.device.index],
                                  pred_box, gt * args.img_size, i)

        #detect_time = _t['im_detect'].toc(average=False)
        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets
        #result = 'im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time)
        #progress.write(result)
        #print(result)

    #progress.close()
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


def visualize_deformation(cfg, img_tensor, deform_pyramid, reg_boxes, default_boxes,
                          pred_boxes, ground_truth, img_idx):
    """
    :param cfg:
    :param img_tensor:
    :param deform_pyramid:
    :param reg_boxes: input is in point form
    :param default_boxes: input is in center form
    :param pred_boxes:
    :param ground_truth:
    :param img_idx:
    :return:
    """
    path = os.path.expanduser("~/Pictures/deform_vis_%s"%args.name)
    img_path = os.path.join(path, "img_%d" % img_idx)
    # reg_boxes to pyramid format:
    start_idx = 0
    reg_boxes_pyramids = []
    prior_pyramids = []
    for i, fm_size in enumerate(cfg["feature_maps"]):
        end_idx = start_idx + fm_size**2 * (2 * len(cfg["aspect_ratios"][i]) + 2)
        reg_boxes_pyramids.append(reg_boxes[:, start_idx:end_idx, :])
        prior_pyramids.append(default_boxes[start_idx:end_idx, :])
        start_idx = end_idx
    if not os.path.exists(path):
        os.mkdir(path)
    height = img_tensor.size(2)
    width = img_tensor.size(3)
    fm_size = cfg['feature_maps'][:len(deform_pyramid)]
    # get deformation maps at different scale
    for i, deform_maps in enumerate(deform_pyramid):
        if i == 0:
            # feature are too small
            continue
        if deform_maps is None:
            continue
        # get deformation maps for different ratio
        per_ratio = []
        #start_idx = 0
        fm_reg_boxes = reg_boxes_pyramids[i]
        fm_priors = prior_pyramids[i]
        for _d, deform in enumerate(deform_maps):
            scale = height / deform.size(2)

            # Get corresponding prior boxes and regressed box
            idx = torch.tensor([_d + len(deform_maps) * i for i in range(deform.size(2) ** 2)]).long()
            priors = fm_priors[idx, :]
            boxes = fm_reg_boxes[:, idx, :]

            if args.rematch:
                overlaps = jaccard(torch.tensor(ground_truth[:, :-1]).float() / args.img_size,
                                   boxes.squeeze(0).data)
            else:
                overlaps = jaccard(torch.tensor(ground_truth[:, :-1]).float() / args.img_size,
                                   point_form(priors))
            try:
                matched_idx = [int(_idx) for _idx in torch.sum(overlaps >= 0.5, dim=0).nonzero().squeeze()]
            except TypeError:
                # Means no priors match the ground truth
                matched_idx = []
            d_x = torch.mean(deform[:, 0::2, :, :], dim=1).unsqueeze(1)
            d_y = torch.mean(deform[:, 1::2, :, :], dim=1).unsqueeze(1)
            deform = torch.cat([d_x, d_y], dim=1)
            if i == 4 and _d == 0:
                z=0
            per_batch = []
            for j in range(deform.size(0)):
                box = boxes[j]
                img = img_tensor[j].permute(1, 2, 0).data.cpu().numpy()
                # Convert to numpy and RGB form
                img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype("uint8")[:, :, (2, 1, 0)].copy()
                dm = deform[j].view(2, -1).data.permute(1, 0).cpu().numpy()
                idx_x = [int(round(num))
                         for num in np.linspace(width / fm_size[i], width, fm_size[i] + 1)][:-1]
                idx_y = [int(round(num))
                         for num in np.linspace(height / fm_size[i], height, fm_size[i] + 1)][:-1]
                assert dm.shape[0] == len(idx_x) * len(idx_y)
                x_coords = idx_x * len(idx_y)
                y_coords = [val for val in idx_y for _ in idx_x]

                # Draw prediction on image
                for pred in pred_boxes:
                    x1, y1, x2, y2 = norm(pred[0]), norm(pred[1]), norm(pred[2]), norm(pred[3])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (94, 218, 250), 2)
                # Draw Ground Truth on image
                for gt in ground_truth:
                    x1, y1, x2, y2 = norm(gt[0]), norm(gt[1]), norm(gt[2]), norm(gt[3])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (33, 235, 63), 2)

                gifs = []
                for k, x in enumerate(x_coords):
                    cv2.line(img, (x, y_coords[k]), (int(round(x + dm[k, 1] * scale)),
                                                      int(round(y_coords[k] + dm[k, 0] * scale))), (0, 0, 255), 1)
                    _img = img.copy()
                    # Draw Priors
                    if k in matched_idx:
                        # purple
                        color = (238, 23, 179)
                        line_width = 2
                    else:
                        # blue
                        color = (255, 105, 65)
                        line_width = 1
                    x1, y1, x2, y2 = coord_to_rect(priors[k], args.img_size, args.img_size)
                    cv2.rectangle(_img, (x1, y1), (x2, y2), color, line_width)
                    # Draw Regressed Boxes
                    x1, y1, x2, y2 = (box[k].clamp(min=0, max=1) * args.img_size).long().cpu().numpy()
                    cv2.rectangle(_img, (x1, y1), (x2, y2), (70, 67, 238), 1)
                    if args.visualize_gif:
                        gifs.append(_img[:, :, [2, 1, 0]])
                per_batch.append(img)
                #for _idx, img in enumerate(gifs):
                    #cv2.imwrite("/home/wang/Pictures/tmp_%s.jpg"%str(_idx).zfill(4), img)
                if not os.path.exists(img_path):
                    os.mkdir(img_path)
                if args.visualize_gif:
                    imageio.mimsave(
                        os.path.join(img_path, 'fm_%s_asr_%s.gif'%(cfg["feature_maps"][i], _d)),
                        gifs)
            per_ratio.append(per_batch)
        per_ratio = list(map(list, zip(*per_ratio)))
        for batch_id, ratio in enumerate(per_ratio):
            odd_ratio = np.concatenate([r for i, r in enumerate(ratio) if i % 2 == 1], axis=1)
            even_ratio = np.concatenate([r for i, r in enumerate(ratio) if i % 2 == 0], axis=1)
            img = np.concatenate([odd_ratio, even_ratio], axis=0)
            name = "iter_%s_fm_%s.jpg"%(str(args.iter).zfill(5), str(fm_size[i]).zfill(2))
            cv2.imwrite(os.path.join(img_path, name), img)


def coord_to_rect(boxes, height, width):
    """
    Convert 4 point boundbox coordinate to matplotlib rectangle coordinate
    """
    x1, y1, x2, y2  = torch.cat((boxes[:2] - boxes[2:] / 2, boxes[:2] + boxes[2:] / 2), 0).clamp_(max=1, min=0)
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

def norm(x):
    x = 0 if x < 0 else x
    x = args.img_size if x >= args.img_size else x
    return int(x)

if __name__ == '__main__':
    # load net
    if args.cuda_id >= torch.cuda.device_count():
        print("argument --cuda_id is larger than the cuda")
        args.cuda_id = 0
    with torch.cuda.device(args.cuda_id):
        num_classes = len(labelmap) + 1                      # +1 for background
        net = build_ssd(args, 'test', args.img_size, num_classes)
        # initialize SSD
        model_name = "%s_%s_%s.pth"%(args.name, args.img_size, args.iter)
        print("Evaluation on model: %s"%model_name)
        pretrained_weight = torch.load(os.path.join(args.save_folder, model_name))
        net.load_state_dict(pretrained_weight)
        net.eval()
        print('Finished loading model!')
        # load data
        dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                               BaseTransform(args.img_size, dataset_mean),
                               VOCAnnotationTransform())
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        # evaluation
        test_net(args.save_folder, net, args.cuda, dataset,
                 BaseTransform(net.size, dataset_mean), args.top_k, args.img_size)
