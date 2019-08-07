from __future__ import division
from itertools import product as product
import torch


class ReceptiveFieldPrior(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg, kernel_size=3):
        super(ReceptiveFieldPrior, self).__init__()
        self.image_size = cfg['min_dim']
        self.kernel_size = kernel_size
        self.feature_maps = cfg['feature_maps']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']

    def __call__(self, *args, **kwargs):
        return self.forward()

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.kernel_size / f_k
                mean += [cx, cy, s_k, s_k] * (2 + 2 * len(self.aspect_ratios[k]))
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        return output


if __name__ == "__main__":
    from .prior_box import PriorBox
    cfg = {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [300 / 38, 300 / 19, 30, 60, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC',
    }
    receptive_field = ReceptiveFieldPrior(cfg)
    boxes = receptive_field()