import torch
import torch.nn as nn
import mmdet.ops.dcn as dcn
from layers.box_utils import *
from layers.visualization import *

class DetectionHeader(nn.Module):
    def __init__(self, in_channel, ratios, num_classes, opt, ):
        super().__init__()
        self.kernel_size = 3
        self.img_size = opt.img_size
        self.opt = opt
        self.deformation = opt.deformation
        self.loc_deformation = opt.loc_deformation

        self.loc_layers = nn.ModuleList([])
        for i in range(ratios):
            if opt.loc_deformation:
                if opt.loc_deform_layer.lower() == "normal":
                    self.loc_layers.append(dcn.DeformConv(in_channel, 4, kernel_size=3, padding=1))
                elif opt.loc_deform_layer.lower() == "incep":
                    self.loc_layers.append(DeformableInception(in_channel, 4, filters=opt.loc_deform_filters,
                                                               inner_groups=len(opt.loc_deform_increment)+1,
                                                               concat_block=opt.concat_block))
                else:
                    raise NotImplementedError()
            else:
                self.loc_layers.append(nn.Conv2d(in_channel, 4, kernel_size=3, padding=1))

        if opt.deformation and \
            opt.deformation_source.lower() not in ["geometric", "geometric_v2", "geometric_v3"]:
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
                                                  kernel_size=self.kernel_size, bias=False,
                                                  concat_block=opt.concat_block)
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
        #if self.loc_deformation:
        regression = []
        for i, loc in enumerate(self.loc_layers):
            idx = torch.tensor([i + len(self.loc_layers) * _
                                for _ in range(x.size(2) * x.size(3))]).long()
            reg_center = prior_centroid[idx, :].repeat(x.size(0), 1).view(x.size(0), x.size(2), x.size(3), -1).permute(0, 3, 1, 2)
            centroid = rf_centroid[idx, :].repeat(x.size(0), 1).view(x.size(0), x.size(2), x.size(3), -1).permute(0, 3, 1, 2)
            df_map = (reg_center - centroid) * x.size(2)
            if self.loc_deformation:
                if self.opt.loc_deform_layer.lower() == "normal":
                    regression.append(loc(x, df_map))
                elif self.opt.loc_deform_layer.lower() == "incep":
                    df_map = [df_map]
                    median = int(reg_center.size(1) / 2)
                    median = reg_center[:, median - 1:median + 1, :, :].repeat(1, self.kernel_size ** 2, 1, 1)
                    for increment in self.opt.cls_deform_increment:
                        # Constrain the extended regression not to exceed the boundary of image
                        new_reg = (median + (reg_center - median) * increment)#.clamp(min=0, max=1)
                        df_map.append((new_reg - centroid) * x.size(2))
                    regression.append(loc(x, df_map))
                else:
                    raise NotImplementedError()
            else:
                regression.append(loc(x))
                df_map = None
            if 10 <= x.size(2) <= 20 and True:
                boxes =decode(regression[-1].permute(0, 2, 3, 1).contiguous().view(-1, 4),
                              priors[idx], cfg["variance"])#.clamp(min=0, max=1)
                visualize_box_and_center(i, centroid, prior=point_form(priors[idx]), reg=boxes, df_map=df_map)
        if verbose:
            print("regression shape is composed of %d %s" % (len(regression), str(regression[0].shape)))
        if self.deformation:
            if self.opt.deformation_source.lower() == "input":
                cls_deform_map = [offset(x) for offset in self.offset_groups]
            elif self.opt.deformation_source.lower() == "regression":
                cls_deform_map = [offset(regression[i]) for i, offset in enumerate(self.offset_groups)]
            elif self.opt.deformation_source.lower() in ["geometric", "geometric_v2", "geometric_v3"]:
                cls_deform_map = []
                for i, reg in enumerate(regression):
                    # get the index of certain ratio from prior box
                    idx = torch.tensor([i + len(regression) * _
                                        for _ in range(reg.size(2) * reg.size(3))]).long()
                    prior = priors[idx, :]
                    if self.opt.deformation_source.lower() == "geometric":
                        centroid = prior_centroid[idx, :].repeat(x.size(0), 1)
                    else:
                        centroid = rf_centroid[idx, :].repeat(x.size(0), 1)
                    _reg = decode(reg.permute(0, 2, 3, 1).contiguous().view(-1, 4),
                                  prior.repeat(x.size(0), 1), cfg["variance"])#.clamp(min=0, max=1)
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

                    reg_center = center_conv_point(_reg,
                                                   v3_form=self.opt.deformation_source.lower() == "geometric_v3")
                    #if x.size(2) == 10:
                        #visualize_box_and_center(
                            #_reg.view(x.size(0), reg.size(2) * reg.size(3), -1)[0],
                            #centroid[idx, :], reg_center.view(x.size(0),
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
                            new_reg = (median + (reg_center - median) * increment)#.clamp(min=0, max=1)
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
        if filters is None or concat_block:
            out_dim = num_classes
        else:
            out_dim = filters
        for i in range(inner_groups):
            self.inner_blocks.append(dcn.DeformConv(in_channel, out_dim,
                                                    kernel_size=kernel_size, padding=1, bias=bias))
        if concat_block:
            self.final_block = nn.Conv2d(inner_groups * out_dim, num_classes,
                                         kernel_size=1, padding=0)

    def forward(self, x, deform_map):
        assert len(deform_map) == len(self.inner_blocks)
        out = [block(x, deform_map[i]) for i, block in enumerate(self.inner_blocks)]
        if self.concat_block:
            out = sum(out) / len(sum)
        else:
            out = self.final_block(torch.cat(out, dim=1))
        return out