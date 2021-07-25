import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, min_holes, max_holes, min_length, max_length):
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        holes = np.random.randint(low=self.min_holes, high=self.max_holes)
        for n in range(holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            l = np.random.randint(low=self.min_length, high=self.max_length)

            y1 = np.clip(y - l // 2, 0, h)
            y2 = np.clip(y + l // 2, 0, h)
            x1 = np.clip(x - l // 2, 0, w)
            x2 = np.clip(x + l // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
