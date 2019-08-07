import os, argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def prepare_args(root_dir):
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Evaluation')

    # Basic Argument
    parser.add_argument('-impl', '--implementation', default="190709", type=str,
                        choices=["vanilla", "header", "190709"],
                        help='vanilla represent default implementation, where conf and loc layers are not divided by ratio;'
                             'In header impl. conf and loc layers are separate by ratio, also support deformation'
                             '170709 on the basis of header, it can add deformation on deep layers')
    parser.add_argument('--save_folder', default='weights', type=str,
                        help='Trained state_dict file path to open')
    parser.add_argument('--iter', default=20000, type=int,
                        help='num of trained iterations')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to train model')
    parser.add_argument('--name', default='SSD', type=str, help='Model name')

    # Dataset
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default=root_dir,
                        help='Dataset root directory path')
    parser.add_argument('--voc_root', default=root_dir,
                        help='Location of VOC root directory')
    parser.add_argument('--year', default=2007, type=int, help='which set to test')
    parser.add_argument('--set', default='test', type=str, help='which set to select',
                        choices=['train', 'val', 'trainval', 'test'])

    # Model Architecture
    parser.add_argument('--img_size', default=300, type=int, help='input image size')
    parser.add_argument('--deformation', default=False, type=str2bool,
                        help='use deformation in detection head')
    parser.add_argument('-kwd', '--kernel_wise_deform', default=False, type=str2bool,
                        help='if True, apply deformation for each pixel in kernel')
    parser.add_argument('--deformation_source', default='concate', type=str,
                        help='the source tensor to generate deformation tensor')
    parser.add_argument('--deform_offset_bias', default=False, type=str2bool,
                        help='allow bias or not')
    parser.add_argument('--deform_offset_dilation', default=1, type=int,
                        help='allow bias or not')
    parser.add_argument("--top_k", type=int, help="detector top_k", default=200)
    parser.add_argument("--conf_threshold", type=float, default=0.01,
                        help="detector_conf_threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.45,
                        help="detector_nms_threshold")
    parser.add_argument('--overlap_threshold', type=float, default=0.5,
                        help='overlap threshold to match prior to ground truth')
    parser.add_argument("--rematch", action="store_true",
                        help="if true, we will use the regressed box produced by localizer to calculate the loss.")


    # Training Parameter
    parser.add_argument('--resume', action="store_true", help="finetuning")
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model')

    parser.add_argument('--ft_iter', default=0, type=int,
                        help='decide which should be loaded')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--max_iter', default=20010, type=int,
                        help='iteration times for training')

    parser.add_argument('--batch_size', default=96, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=6, type=int,
                        help='Number of workers used in dataloading')

    # Optimizer
    parser.add_argument('--optimizer', default="adam", type=str, choices=["adam", "sgd"],
                        help='which optimizer to use.')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')


    # Evaluation Parameter
    parser.add_argument('-vd', '--visualize_deformation', action="store_true",
                        help="visualize deformation or not")
    parser.add_argument('--visualize_gif', action="store_true",
                        help="visualize detection process as gif or not")
    parser.add_argument('--cuda_id', default=2, type=int,
                        help='device id of test')

    # Others
    parser.add_argument('--visdom', default=False, type=str2bool,
                        help='Use visdom for loss visualization')
    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    return args