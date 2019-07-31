import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from layers.box_utils import *
from data import voc, coco
import os
import mmdet.ops.dcn as dcn

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
        self.priorbox = PriorBox(self.cfg)
        prior = self.priorbox.forward()
        self.priors = [prior.cuda(i) for i in range(torch.cuda.device_count())]
        self.prior_centeroids = [center_conv_point(point_form(prior).clamp(min=0, max=1))
                                 for prior in self.priors]
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

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, bkg_label=0, top_k=args.top_k,
                                 conf_thresh=args.conf_threshold, nms_thresh=args.nms_threshold)

    def forward(self, x, deform_map=False):
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

        # apply multibox head to source layers
        if self.args.implementation in ["header", "190709"]:
            start_id = 0
            for idx, (x, h) in enumerate(zip(sources, self.header)):
                end_id = start_id + x.size(2) * x.size(3) * (2 + 2 * len(self.cfg["aspect_ratios"][idx]))
                if deform_map:
                  l, c, d = h(x, input_h, deform_map=deform_map, priors=self.priors[x.device.index][start_id: end_id],
                              prior_centeroids=self.prior_centeroids[x.device.index][start_id: end_id], cfg=self.cfg)
                  deform.append(d)
                else:
                  l, c = h(x, input_h, deform_map=deform_map, priors=self.priors[x.device.index][start_id: end_id],
                           prior_centeroids=self.prior_centeroids[x.device.index][start_id: end_id], cfg=self.cfg)
                start_id = end_id
                loc.append(l.permute(0, 2, 3, 1).contiguous())
                conf.append(c.permute(0, 2, 3, 1).contiguous())
        elif self.args.implementation == "vanilla":
            for (x, l, c) in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors[x.device.index].type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        if deform_map:
            return output, deform
        else:
            return output

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
            loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        return vgg, extra_layers, (loc_layers, conf_layers)
    else:
        raise NotImplementedError()


class DetectionHeader(nn.Module):
    def __init__(self, in_channel, ratios, num_classes, opt):
        super().__init__()
        self.kernel_wise_deform = opt.kernel_wise_deform
        self.deformation_source = opt.deformation_source
        self.kernel_size = 3
        self.deformation = opt.deformation

        self.loc_layers = nn.ModuleList([])
        for i in range(ratios):
            self.loc_layers.append(nn.Conv2d(in_channel, 4, kernel_size=3, padding=1))

        if opt.deformation and opt.deformation_source.lower() != "geometric":
            self.offset_groups = nn.ModuleList([])
            if opt.deformation_source.lower() == "input":
                # Previous version, represent deformation_source is True
                offset_in_channel = in_channel
            elif opt.deformation_source.lower() == "regression":
                # Previous version, represent deformation_source is False
                offset_in_channel = 4
            elif opt.deformation_source.lower() == "concate":
                offset_in_channel = in_channel + 4
            elif opt.deformation_source.lower() == "geometric":
                raise ArithmeticError()
            else:
                raise NotImplementedError()
            if opt.kernel_wise_deform:
                deform_depth = 2
            else:
                deform_depth = 2 * (self.kernel_size ** 2)
            for i in range(ratios):
                pad = int(0.5 * (self.kernel_size - 1) + opt.deform_offset_dilation - 1)
                _offset2d = nn.Conv2d(offset_in_channel, deform_depth, kernel_size=self.kernel_size,
                                      bias=opt.deform_offset_bias, padding=pad,
                                      dilation=opt.deform_offset_dilation)
                self.offset_groups.append(_offset2d)

        self.conf_layers = nn.ModuleList([])
        for i in range(ratios):
            if opt.deformation:
                _deform = dcn.DeformConv(in_channel, num_classes, kernel_size=self.kernel_size, padding=1, bias=False)
            else:
                _deform = nn.Conv2d(in_channel, num_classes, kernel_size=3, padding=1)
            self.conf_layers.append(_deform)

    def forward(self, x, h, verbose=False, deform_map=False, priors=None, prior_centeroids=None, cfg=None):
        # regression is a list, the length of regression equals to the number different aspect ratio
        # under current receptive field, elements of regression are PyTorch Tensor, encoded in
        # point-form, represent the regressed prior boxes.
        regression = [loc(x) for loc in self.loc_layers]
        if verbose:
            print("regression shape is composed of %d %s" % (len(regression), str(regression[0].shape)))
        if self.deformation:
            self.deformation_source = "geometric"
            if self.deformation_source.lower() == "input":
                _deform_map = [offset(x) for offset in self.offset_groups]
            elif self.deformation_source.lower() == "regression":
                _deform_map = [offset(regression[i]) for i, offset in enumerate(self.offset_groups)]
            elif self.deformation_source.lower() == "geometric":
                if priors is None:
                    raise TypeError("prior should not be none if the deformation source is geometric")
                _deform_map = []
                for i, reg in enumerate(regression):
                    idx = torch.tensor([i + len(regression) * _ for _ in range(reg.size(2) * reg.size(3))]).long()
                    prior = priors[idx, :]
                    prior_center = prior_centeroids[idx, :].repeat(x.size(0), 1)
                    _reg = decode(reg.permute(0, 2, 3, 1).contiguous().view(-1, 4),
                                  prior.repeat(x.size(0), 1), cfg["variance"]).clamp(min=0, max=1)
                    reg_center = center_conv_point(_reg)
                    # print(_reg[0, :].data, point_form(prior[0:1, :]).clamp(min=0, max=1).data)
                    # TODO: In the future work, when input image is not square, we need
                    # TODO: to multiply image with its both width and height
                    df_map = (reg_center - prior_center) * x.size(2)
                    _deform_map.append(df_map.view(x.size(0), reg.size(2), reg.size(3), -1)
                                       .permute(0, 3, 1, 2))
            elif self.deformation_source.lower() == "concate":
                # TODO: reimplement forward graph
                raise NotImplementedError()
            else:
                raise NotImplementedError()

            if verbose:
                print("deform_map shape is composed of %d %s" % (len(_deform_map), str(_deform_map[0].shape)))
            if self.kernel_wise_deform:
                _deform_map = [dm.repeat(1, self.kernel_size ** 2, 1, 1) for dm in _deform_map]
            # Amplify the offset signal, so it can deform the kernel to adjacent anchor
            #_deform_map = [dm * h/x.size(2) for dm in _deform_map]
            if verbose:
                print("deform_map shape is extended to %d %s" % (len(_deform_map), str(_deform_map[0].shape)))
            pred = [deform(x, _deform_map[i]) for i, deform in enumerate(self.conf_layers)]
        else:
            pred = [conf(x) for conf in self.conf_layers]
            _deform_map = None
        if verbose:
            print("pred shape is composed of %d %s" % (len(pred), str(pred[0].shape)))
        if deform_map:
            return torch.cat(regression, dim=1), torch.cat(pred, dim=1), _deform_map
        else:
            return torch.cat(regression, dim=1), torch.cat(pred, dim=1)


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
