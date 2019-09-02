import os, torch, warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.append(os.path.expanduser("~/Documents"))
import omni_torch.visualize.basic as vb
from matplotlib import gridspec
from layers.box_utils import *
import imageio


def print_box(red_boxes=(), shape=0, green_boxes=(), blue_boxes=(), img=None,
              idx=None, title=None, step_by_step_r=False, step_by_step_g=False,
              step_by_step_b=False, name_prefix=None, save_dir=None):
    # Generate the save folder and image save name
    if not name_prefix:
        name_prefix = "tmp"
    if idx is not None:
        img_name = name_prefix + "_sample_%s_pred" % (idx)
    else:
        img_name = name_prefix
    if save_dir is None:
        save_dir = os.path.expanduser("~/Pictures")
    else:
        save_dir = os.path.expanduser(save_dir)
    if not os.path.exists(save_dir):
        warnings.warn(
            "The save_dir you specified (%s) does not exist, saving results under "
            "~/Pictures" % (save_dir)
        )
        save_dir = os.path.expanduser("~/Pictures")
    img_path = os.path.join(save_dir, img_name)

    # Figure out the shape
    if type(shape) is tuple:
        h, w = shape[0], shape[1]
    else:
        h, w = shape, shape
    # img as white background image
    if img is None:
        img = np.zeros((h, w, 3)).astype(np.uint8) + 254
    else:
        img = img.astype(np.uint8)
        h, w, c = img.shape

    # Perform Visualization of boundbox
    fig, ax = plt.subplots(figsize=(round(w / 100), round(h / 100)))
    ax.imshow(img)
    step = 0
    for box in red_boxes:
        x1, y1, x2, y2 = coord_to_rect(box, h, w)
        rect = patches.Rectangle((x1, y1), x2, y2, linewidth=1,
                                 edgecolor='r', facecolor='none', alpha=1)
        ax.add_patch(rect)
        if step_by_step_r:
            plt.savefig(img_path + "_red_step_%s.jpg" % (str(step).zfill(4)))
            step += 1
    for box in green_boxes:
        x1, y1, x2, y2 = coord_to_rect(box, h, w)
        rect = patches.Rectangle((x1, y1), x2, y2, linewidth=2,
                                 edgecolor='g', facecolor='none', alpha=0.4)
        ax.add_patch(rect)
        if step_by_step_g:
            plt.savefig(img_path + "_green_step_%s.jpg" % (str(step).zfill(4)))
            step += 1
    for box in blue_boxes:
        x1, y1, x2, y2 = coord_to_rect(box, h, w)
        rect = patches.Rectangle((x1, y1), x2, y2, linewidth=2,
                                 edgecolor='b', facecolor='none', alpha=0.4)
        ax.add_patch(rect)
        if step_by_step_b:
            plt.savefig(img_path + "_blue_step_%s.jpg" % (str(step).zfill(4)))
            step += 1
    if title:
        plt.title(title)
    plt.savefig(os.path.join(save_dir, img_name + ".jpg"))
    plt.close()


def visualize_overlaps(args, cfg, target, label, prior, after_reg=False):
    images, subtitle, coords = [], [], []

    # conf中的1代表所有当前设置下与ground truth匹配的default box及其相应的index
    if after_reg:
        overlaps, conf = match(args.rematch_overlap_threshold, target, prior,
                               None, label, None, None, 0, visualize=True)
    else:
        overlaps, conf = match(args.overlap_threshold, target, prior,
                               None, label, None, None, 0, visualize=True)
    summary = "%s of %s positive samples" % (int(torch.sum(conf != 0)), prior.size(0))
    crop_start = 0

    #  减1是因为feature map尺寸为1时不做visualization
    for k in range(len(cfg['feature_maps']) - 1):
        # Get the setting from cfg to calculate number of anchor and prior boxes
        h, w = get_parameter(cfg['feature_maps'][k])
        h_stride, w_stride = get_parameter(cfg['stride'][k])
        anchor_num = calculate_anchor_number(cfg, k)
        prior_num = len(range(0, int(h), int(h_stride))) * \
                    len(range(0, int(w), int(w_stride))) * anchor_num

        # Get the index of matched prior boxes and collect these boxes
        _conf = conf[crop_start: crop_start + prior_num]
        _overlaps = overlaps[crop_start: crop_start + prior_num]
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
        best_prior_idx = best_prior_idx.squeeze(1)

        # points = point_form(prior[best_prior_idx].cpu(), ratio)
        # print_box(target, shape=? blue_boxes=points, step_by_step_b=True)

        matched_priors = int(torch.sum(_conf > 0))
        idx = _conf > 0
        idx = list(np.where(idx.cpu().numpy() == 1)[0])
        for i in idx:
            coords.append(point_form(prior[crop_start + i:crop_start + i + 1, :]).squeeze())

        # Reshape _conf into the shape of image so as to visualize it
        _conf = _conf.view(len(range(0, int(h), int(h_stride))),
                           len(range(0, int(w), int(w_stride))), anchor_num)
        _conf = _conf.permute(2, 0, 1)
        ratios = [1, 1] + [_ for _ in cfg['aspect_ratios'][k]] + [1/_ for _ in cfg['aspect_ratios'][k]]
        subs = ["ratio: %s" % (r) for r in ratios]
        subtitle.append("box height: %s\neffective samle: %s"
                        % (cfg['min_sizes'][k], matched_priors))
        #if cfg['big_box']:
            #subs += ["ratio: %s" % (r) for r in cfg['box_ratios_large'][k]]
            #subtitle[-1] = "box height: %s and %s\neffective samle: %s" \
                           #% (cfg['box_height'][k], cfg['box_height_large'][k], matched_priors)

        # Convert _conf into open-cv form
        image = vb.plot_tensor(None, _conf.unsqueeze_(1) * 254, deNormalize=False,
                               sub_title=subs)
        images.append(image.astype(np.uint8))
        crop_start += prior_num
    return images, summary, subtitle, coords


def visualize_bbox(args, cfg, images, targets, prior=None, idx=0, prefix="", start_idx=0, path=None):
    print("Visualizing bound box...")
    ratios = images.size(3) / images.size(2)
    batch = images.size(0)
    height, width = images.size(2) / 100 + 1, images.size(3) / 50 + 1
    for i in range(batch):
        image = images[i].permute(1, 2, 0).data.cpu().numpy()
        image = ((image - np.min(image)) / (np.max(image) - np.min(image))).copy() * 255
        bbox = targets[i]

        #image = vb.plot_tensor(args, image, deNormalize=True, margin=0).astype("uint8")
        h, w = image.shape[0], image.shape[1]
        # Create a Rectangle patch
        rects = []
        for point in bbox:
            x1, y1, x2, y2 =  (point * 300).int().tolist()[:-1]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #rects.append(patches.Rectangle((x1, y1), x2, y2, linewidth=2,
                                           #edgecolor='r', facecolor='none'))
        if prior is not None:
            after_reg = True if prefix is not '' else False
            overlaps, summary, subtitle, coords = \
                visualize_overlaps(args, cfg, bbox[:, :-1].data, bbox[:, -1].data, prior, after_reg=after_reg)
            for coord in coords:
                x1, y1, x2, y2 =  (coord * 300).int().tolist()
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
                #rects.append(patches.Rectangle((x1, y1), x2, y2, linewidth=1,
                                               #edgecolor='b', facecolor='none', alpha=0.3))
        else:
            overlaps = []
            summary = ""
        fig, ax = plt.subplots(figsize=(width + len(overlaps), height))
        width_ratio = [2] + [1] * len(overlaps)
        gs = gridspec.GridSpec(1, 1 + len(overlaps), width_ratios=width_ratio)
        try:
            ax0 = plt.subplot(gs[0])
        except KeyError:
            return
        ax0.imshow(image / 255)
        ax0.set_title(summary)
        for j in range(len(overlaps)):
            ax = plt.subplot(gs[j + 1])
            ax.imshow(overlaps[j])
            ax.set_title(subtitle[j])
        for rect in rects:
            ax0.add_patch(rect)
        plt.grid(False)
        plt.tight_layout()
        if not prefix:
            _prefix = "sample"
        else:
            _prefix = prefix
        if path is None:
            path = args.val_log
        plt.savefig(os.path.join(path, "batch_%s_%s_vis_%s.jpg" % (idx, _prefix, start_idx + i)))
        plt.close()

def visualize_box_and_center(idx, rf_centeroid, prior=None, reg=None,
                             prior_centroid=None, df_map=None, img_size=300):
    """
    :param prior: shape=(?, 4)
    :param rf_centeroid: shape=(?, 18)
    :return:
    """
    h, w = rf_centeroid.size(2), rf_centeroid.size(2)
    bg = cv2.imread("/home/wang/Pictures/tmp.jpg")
    if bg is None:
        bg = np.ones((img_size, img_size, 3)) * 255
    else:
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    if prior is not None:
        prior = prior * img_size
    if reg is not None:
        reg = reg * img_size
    if rf_centeroid is not None:
        rf_centeroid = (rf_centeroid.permute(0, 2, 3, 1).view(rf_centeroid.size(0), -1,
                                                              rf_centeroid.size(1)) * img_size).long().squeeze(0)
    if prior_centroid is not None:
        prior_centroid = (prior_centroid.permute(0, 2, 3, 1).view(prior_centroid.size(0), -1,
                                                                  prior_centroid.size(1)) * prior_centroid).long().squeeze(0)
    if df_map is not None:
        if type(df_map) is not list:
            df_map = [df_map]
        df_map = [rf_centeroid + (df.permute(0, 2, 3, 1).view(df.size(0), -1, df.size(1)) * img_size / h).long().squeeze(0)
                  for df in df_map]

    gif = []
    for i in range(rf_centeroid.size(0)):
        canvas = np.ones((img_size, img_size, 3)) * 128
        if prior is not None:
            x1, y1, x2, y2 = prior[i].tolist()
            cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
        if reg is not None:
            x1, y1, x2, y2 = reg[i].tolist()
            cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (0, 0, 255), 1)

        points = rf_centeroid[i].view(-1, 2).tolist()
        for point in points:
            cv2.circle(canvas, (round(point[1]), round(point[0])), 2, (255, 0, 0), 2)

        if prior_centroid is not None:
            points = prior_centroid[i].view(-1, 2).tolist()
            for point in points:
                cv2.circle(canvas, (round(point[1]), round(point[0])), 2, (0, 255, 0), 2)

        if df_map is not None:
            for df in df_map:
                reg_points = df[i].view(-1, 2).tolist()
                for point in reg_points:
                    cv2.circle(canvas, (round(point[1]), round(point[0])), 2, (0, 0, 255), 2)
        if bg is None:
            gif.append(bg.astype(np.uint8))
        else:
            gif.append((bg * 0.5 + canvas * 0.5).astype(np.uint8))
    imageio.mimsave("/home/wang/Pictures/fm_%s_ratio_%s.gif" %
                    (rf_centeroid.size(0), str(idx).zfill(2)), gif)
    #cv2.imwrite("/home/wang/Pictures/tmp_%s.jpg"%str(i).zfill(3), canvas)