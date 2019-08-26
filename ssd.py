import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from layers.box_utils import *
from data import voc, coco
import os
import mmdet.ops.dcn as dcn
import numpy as np
import cv2
import imageio

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, args, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        priorBox = PriorBox(self.cfg)
        rf_Box = ReceptiveFieldPrior(self.cfg)
        prior = priorBox.forward()
        rf_prior = rf_Box.forward()
        self.priors = [prior.cuda(i) for i in range(torch.cuda.device_count())]
        self.rf_priors = [rf_prior.cuda(i) for i in range(torch.cuda.device_count())]
        self.prior_centeroids = [center_conv_point(point_form(prior).clamp(min=0, max=1))
                                 for prior in self.priors]
        self.rf_prior_centeroids = [center_conv_point(point_form(rf_prior))
                                 for rf_prior in self.rf_priors]
        self.size = size
        self.args = args
        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        if args.implementation in ["header", "190709"]:
            self.header = nn.ModuleList(head)
        elif args.implementation == "vanilla":
            self.loc = nn.ModuleList(head[0])
            self.conf = nn.ModuleList(head[1])
        self.criterion = MultiBoxLoss(self.cfg['num_classes'], args, True, 0,
                                 True, 3, 0.5, False, args.cuda, rematch=args.rematch)
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, bkg_label=0, top_k=args.top_k,
                             conf_thresh=args.conf_threshold, nms_thresh=args.nms_threshold)

    def forward(self, input, y=None, y_idx=None, deform_map=False, test=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        deform = list()

        x = input
        input_h, input_w = x.size(2), x.size(3)
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        #for i in range(self.args.cascade):
        # apply multibox head to source layers
        if self.args.implementation in ["header", "190709"]:
            start_id = 0
            for idx, (x, h) in enumerate(zip(sources, self.header)):
                end_id = start_id + x.size(2) * x.size(3) * (2 + 2 * len(self.cfg["aspect_ratios"][idx]))
                if deform_map:
                  l, c, d = h(x, input_h, deform_map=deform_map,
                              priors=self.priors[x.device.index][start_id: end_id],
                              rf_centroid=self.rf_prior_centeroids[x.device.index][start_id: end_id],
                              centroid=self.prior_centeroids[x.device.index][start_id: end_id],
                              cfg=self.cfg, y=y)
                  deform.append(d)
                else:
                  l, c = h(x, input_h, deform_map=deform_map,
                           priors=self.priors[x.device.index][start_id: end_id],
                           rf_centroid=self.rf_prior_centeroids[x.device.index][start_id: end_id],
                           prior_centroid=self.prior_centeroids[x.device.index][start_id: end_id],
                           cfg=self.cfg, y=y)
                start_id = end_id
                loc.append(l.permute(0, 2, 3, 1).contiguous())
                conf.append(c.permute(0, 2, 3, 1).contiguous())
        elif self.args.implementation == "vanilla":
            for (x, l, c) in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        else:
            raise NotImplementedError()

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if test:
            output = self.detect(loc.view(loc.size(0), -1, 4),
                                 self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                                 self.priors[x.device.index].type(type(x.data)))
            return output, deform
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
            loss_l, loss_c = self.criterion(output, y, y_idx, images=input)
            #print(loss_l, loss_c)
            return loss_l.unsqueeze(0), loss_c.unsqueeze(0)


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

#cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256], i=1024
def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes, opt):
    header = []
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    if opt.implementation in ["header", "190709"]:
        for k, v in enumerate(vgg_source):
            header += [DetectionHeader(vgg[v].out_channels, cfg[k], num_classes, opt)]
        for k, v in enumerate(extra_layers[1::2], 2):
            if opt.implementation == "190709":
                if k > 4:
                    opt.deformation = False
            else:
                opt.deformation = False
            header += [DetectionHeader(v.out_channels, cfg[k], num_classes, opt)]
        return vgg, extra_layers, header
    elif opt.implementation == "vanilla":
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4,
                                     kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes,
                                      kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4,
                                     kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes,
                                      kernel_size=3, padding=1)]
        return vgg, extra_layers, (loc_layers, conf_layers)
    else:
        raise NotImplementedError()


class DetectionHeader(nn.Module):
    def __init__(self, in_channel, ratios, num_classes, opt, ):
        super().__init__()
        self.kernel_size = 3
        self.img_size = opt.img_size
        self.opt = opt
        self.deformation = opt.deformation

        self.loc_layers = nn.ModuleList([])
        for i in range(ratios):
            if opt.loc_deformation:
                if opt.loc_deform_layer.lower() == "normal":
                    self.loc_layers.append(nn.Conv2d(in_channel, 4, kernel_size=3, padding=1))
                elif opt.loc_deform_layer.lower() == "incep":
                    self.loc_layers.append(DeformableInception(in_channel, 4, filters=opt.loc_deform_filters,
                                                               inner_groups=len(opt.loc_deform_increment)+1,))
                else:
                    raise NotImplementedError()
            else:
                self.loc_layers.append(nn.Conv2d(in_channel, 4, kernel_size=3, padding=1))

        if opt.deformation and \
            opt.deformation_source.lower() not in ["geometric", "geometric_v2"]:
            self.offset_groups = nn.ModuleList([])
            if opt.deformation_source.lower() == "input":
                # Previous version, represent deformation_source is True
                offset_in_channel = in_channel
            elif opt.deformation_source.lower() == "regression":
                # Previous version, represent deformation_source is False
                offset_in_channel = 4
            elif opt.deformation_source.lower() == "concate":
                offset_in_channel = in_channel + 4
            else:
                raise NotImplementedError()
            if opt.kernel_wise_deform:
                deform_depth = 2
            else:
                deform_depth = 2 * (self.kernel_size ** 2)
            for i in range(ratios):
                pad = int(0.5 * (self.kernel_size - 1) + opt.deform_offset_dilation - 1)
                _offset2d = nn.Conv2d(offset_in_channel, deform_depth,
                                      kernel_size=self.kernel_size,
                                      bias=opt.deform_offset_bias, padding=pad,
                                      dilation=opt.deform_offset_dilation)
                self.offset_groups.append(_offset2d)

        self.conf_layers = nn.ModuleList([])
        for i in range(ratios):
            if opt.deformation:
                if opt.cls_deform_layer.lower() == "normal":
                    _deform = dcn.DeformConv(in_channel, num_classes,
                                             kernel_size=self.kernel_size,
                                             padding=1, bias=False)
                elif opt.cls_deform_layer.lower() == "incep":
                    _deform = DeformableInception(in_channel, num_classes, filters=opt.cls_deform_filters,
                                                  inner_groups=len(opt.cls_deform_increment)+1,
                                                  kernel_size=self.kernel_size, bias=False)
                else:
                    raise NotImplementedError()
            else:
                _deform = nn.Conv2d(in_channel, num_classes, kernel_size=3, padding=1)
            self.conf_layers.append(_deform)

    def forward(self, x, h, verbose=False, deform_map=False, priors=None, prior_centroid=None,
                rf_centroid=None, cfg=None, y=None):
        # regression is a list, the length of regression equals to the number different aspect ratio
        # under current receptive field, elements of regression are PyTorch Tensor, encoded in
        # point-form, represent the regressed prior boxes.
        # regression = [loc(x) for loc in self.loc_layers]
        regression = []
        loc_deform_map = []
        for i, loc in enumerate(self.loc_layers):
            idx = torch.tensor([i + len(self.loc_layers) * _
                                for _ in range(x.size(2) * x.size(3))]).long()
            reg_center = prior_centroid[idx, :].repeat(x.size(0), 1)
            centroid = rf_centroid[idx, :].repeat(x.size(0), 1)
            df_map = (reg_center - centroid) * x.size(2)
            df_map = df_map.view(x.size(0), x.size(2), x.size(3), -1).permute(0, 3, 1, 2)
            if self.opt.cls_deform_layer.lower() == "normal":
                loc_deform_map.append(df_map)
            elif self.opt.cls_deform_layer.lower() == "incep":
                df_map = [df_map]
                median = int(reg_center.size(1) / 2)
                median = reg_center[:, median - 1:median + 1, :, :].repeat(1, self.kernel_size ** 2, 1, 1)
                for increment in self.opt.cls_deform_increment:
                    # Constrain the extended regression not to exceed the boundary of image
                    new_reg = (median + (reg_center - median) * increment).clamp(min=0, max=1)
                    df_map.append((new_reg - centroid) * x.size(2))
                    loc_deform_map.append(df_map)
            else:
                raise NotImplementedError()



        if verbose:
            print("regression shape is composed of %d %s" % (len(regression), str(regression[0].shape)))
        if self.deformation:
            if self.opt.deformation_source.lower() == "input":
                cls_deform_map = [offset(x) for offset in self.offset_groups]
            elif self.opt.deformation_source.lower() == "regression":
                cls_deform_map = [offset(regression[i]) for i, offset in enumerate(self.offset_groups)]
            elif self.opt.deformation_source.lower() in ["geometric", "geometric_v2"]:
                cls_deform_map = []
                for i, reg in enumerate(regression):
                    # get the index of certain ratio from prior box
                    idx = torch.tensor([i + len(regression) * _
                                        for _ in range(reg.size(2) * reg.size(3))]).long()
                    prior = priors[idx, :]
                    centroid = rf_centroid[idx, :].repeat(x.size(0), 1)
                    _reg = decode(reg.permute(0, 2, 3, 1).contiguous().view(-1, 4),
                                  prior.repeat(x.size(0), 1), cfg["variance"]).clamp(min=0, max=1)
                    if self.opt.gt_replace:
                        overlaps = jaccard(y[:, :-1], _reg)
                        tmp = (overlaps > 0.6).nonzero().tolist()
                        for t in tmp:
                            _reg[t[1]] = y[t[0], :-1]
                            reg_i = t[1] // x.size(2)
                            reg_j = t[1] - (reg_i * x.size(2))
                            regression[i][:, :, reg_i, reg_j] = \
                                encode(y[t[0]:t[0]+1, :-1], prior[t[1]:t[1]+1], cfg["variance"])
                            #decode(regression[i][:, :, 11, 4], prior[t[1]].unsqueeze(0), cfg["variance"])

                    reg_center = center_conv_point(_reg)
                    #if 1 < x.size(2) <= 10:
                        #visualize_box_and_center(
                            #_reg.view(x.size(0), reg.size(2) * reg.size(3), -1)[0],
                            #centeroids[idx, :], reg_center.view(x.size(0),
                            #reg.size(2) * reg.size(3), -1)[0], i)
                    # print(_reg[0, :].data, point_form(prior[0:1, :]).clamp(min=0, max=1).data)
                    # TODO: In the future work, when input image is not square, we need
                    # TODO: to multiply image with its both width and height
                    reg_center = reg_center.view(x.size(0), reg.size(2), reg.size(3), -1).permute(0, 3, 1, 2)
                    centroid = centroid.view(x.size(0), reg.size(2), reg.size(3), -1).permute(0, 3, 1, 2)
                    df_map = (reg_center - centroid) * x.size(2)
                    if self.opt.cls_deform_layer.lower() == "normal":
                        cls_deform_map.append(df_map)
                    elif self.opt.cls_deform_layer.lower() == "incep":
                        df_map = [df_map]
                        median = int(reg_center.size(1) / 2)
                        median = reg_center[:, median-1:median+1, :, :].repeat(1, self.kernel_size**2, 1, 1)
                        for increment in self.opt.cls_deform_increment:
                            # Constrain the extended regression not to exceed the boundary of image
                            new_reg = (median + (reg_center - median) * increment).clamp(min=0, max=1)
                            df_map.append((new_reg - centroid) * x.size(2))
                        cls_deform_map.append(df_map)
                    else:
                        raise NotImplementedError()
            elif self.opt.deformation_source.lower() == "concate":
                raise NotImplementedError()
            else:
                raise NotImplementedError()

            if verbose:
                print("deform_map shape is composed of %d %s" % (len(cls_deform_map), str(cls_deform_map[0].shape)))
            if self.opt.kernel_wise_deform:
                cls_deform_map = [dm.repeat(1, self.kernel_size ** 2, 1, 1) for dm in cls_deform_map]
                if verbose:
                    print("deform_map shape is extended to %d %s" % (len(cls_deform_map), str(cls_deform_map[0].shape)))
            pred = [deform(x, cls_deform_map[i]) for i, deform in enumerate(self.conf_layers)]
        else:
            pred = [conf(x) for conf in self.conf_layers]
            cls_deform_map = None
        if verbose:
            print("pred shape is composed of %d %s" % (len(pred), str(pred[0].shape)))
        if deform_map:
            return torch.cat(regression, dim=1), torch.cat(pred, dim=1), cls_deform_map
        else:
            return torch.cat(regression, dim=1), torch.cat(pred, dim=1)


class DeformableInception(nn.Module):
    def __init__(self, in_channel, num_classes, inner_groups=2, filters=None, kernel_size=3,
                 bias=False, concat_block=False):
        super().__init__()
        self.inner_blocks = nn.ModuleList([])
        self.concat_block = concat_block
        if filters:
            out_dim = filters
        else:
            out_dim = num_classes
        for i in range(inner_groups):
            self.inner_blocks.append(dcn.DeformConv(in_channel, out_dim,
                                                    kernel_size=kernel_size, padding=1, bias=bias))
        if concat_block:
            self.concat_block = nn.Conv2d(inner_groups * out_dim, inner_groups * out_dim,
                                         kernel_size=1, padding=0)
        self.final_block = nn.Conv2d(inner_groups * out_dim, num_classes,
                                     kernel_size=kernel_size, padding=1)

    def forward(self, x, deform_map):
        assert len(deform_map) == len(self.inner_blocks)
        out = [block(x, deform_map[i]) for i, block in enumerate(self.inner_blocks)]
        if self.concat_block:
            out = self.concat_block(torch.cat(out, dim=1))
            out = self.final_block(out)
        else:
            out = self.final_block(torch.cat(out, dim=1))
        return out

def visualize_box_and_center(box, centeroid, reg_center, idx, img_size=300):
    """
    :param box: shape=(?, 4)
    :param centeroid: shape=(?, 18)
    :return:
    """
    bg = cv2.imread("/home/wang/Pictures/tmp.jpg")
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    box = box * img_size
    centeroid = centeroid * img_size
    reg_center = reg_center * img_size
    gif = []
    for i in range(box.size(0)):
        canvas = np.ones((img_size, img_size, 3)) * 200
        x1, y1, x2, y2 = box[i].tolist()
        #print("img: %s"%str(i).zfill(3))
        #print(centeroid[i])
        #print(reg_center[i])
        points = centeroid[i].view(-1, 2).tolist()
        reg_points = reg_center[i].view(-1, 2).tolist()
        cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (255, 0, 0), 2)
        for point in points:
            cv2.circle(canvas, (round(point[0]), round(point[1])), 3, (0, 0, 255), 2)
        for point in reg_points:
            cv2.circle(canvas, (round(point[0]), round(point[1])), 3, (0, 255, 0), 2)
        gif.append((bg * 0.5 + canvas * 0.5).astype(np.uint8))
    imageio.mimsave("/home/wang/Pictures/fm_%s_ratio_%s.gif"%(box.size(0), str(idx).zfill(2)), gif)
    #cv2.imwrite("/home/wang/Pictures/tmp_%s.jpg"%str(i).zfill(3), canvas)

class FPN(nn.Module):
    def __init__(self):
        super().__init__()


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(opt, phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg=vgg(base[str(size)], 3),
                                     extra_layers=add_extras(extras[str(size)], 1024),
                                     cfg=mbox[str(size)], num_classes=num_classes, opt=opt)
    return SSD(opt, phase, size, base_, extras_, head_, num_classes)
