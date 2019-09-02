import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp
from layers.box_utils import *
from ..visualization import *


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, args, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, rematch=False):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.args = args
        self.threshold = args.overlap_threshold
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.rematch = rematch

    def forward(self, predictions, targets, targets_idx, images=None):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        batch_num = loc_data.size(0)
        priors = priors[loc_data.device.index][:loc_data.size(1), :]
        num_priors = (priors.size(0))

        targets_idx = targets_idx.tolist()
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.cuda.FloatTensor(batch_num, num_priors, 4)
        conf_t = torch.cuda.LongTensor(batch_num, num_priors)

        for idx in range(batch_num):
            truths = targets[targets_idx[idx][0]: targets_idx[idx][0] + targets_idx[idx][1], :-1].data
            labels = targets[targets_idx[idx][0]: targets_idx[idx][0] + targets_idx[idx][1], -1].data
            #truths = targets[idx][:, :-1].data
            #labels = targets[idx][:, -1].data
            if self.args.rematch > 0 and self.args.curr_epoch > self.args.rematch:
            #if self.rematch:
                defaults = center_size(decode(loc_data[idx], priors.data, self.variance))#.clamp(min=0, max=1))
                if self.args.visualize_box:
                    start_idx = priors.device.index * batch_num + idx
                    _target = targets[targets_idx[idx][0]: targets_idx[idx][0] + targets_idx[idx][1], :].data
                    visualize_bbox(self.args, cfg, images[idx:idx+1], [_target], defaults, 0, prefix="reg", start_idx=start_idx)
                    pass
                threshold = self.args.rematch_overlap_threshold
            else:
                if self.args.visualize_box:
                    start_idx = priors.device.index * batch_num + idx
                    _target = targets[targets_idx[idx][0]: targets_idx[idx][0] + targets_idx[idx][1], :].data
                    visualize_bbox(self.args, cfg, images[idx:idx+1], [_target], defaults, 0, prefix="reg", start_idx=start_idx)
                defaults = priors.data
                threshold = self.threshold
            match(threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        # wrap targets
        #loc_t = Variable(loc_t, requires_grad=False)
        #conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # batch_conf.gather(1, conf_t.view(-1, 1))
        # 将所有
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(batch_num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
