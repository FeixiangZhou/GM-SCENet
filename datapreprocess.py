import glob
import os
import torch
import cv2
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import matplotlib.pyplot as plt
import imageio

# tranforms
class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [-1,1]."""
       #Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
       #output[channel] = (input[channel] - mean[channel]) / std[channel]

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        for i in range(image.shape[0]):
            image[i,:,:] = (image[i,:,:] -0.5) / 0.5

        return {'image': image, 'keypoints': key_pts}


class Rescale(object):
    """Rescale the image in a sample to a given size.e

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']
        # images = imageio.imread(image)



        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        # heat = heatmaps.transpose((1, 2, 0))
        # heat = cv2.resize(heat, (new_w, new_h))
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class Aug(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        # ia.seed(1)
        kps = KeypointsOnImage([
            Keypoint(x=key_pts[0][0], y=key_pts[0][1]),
            Keypoint(x=key_pts[1][0], y=key_pts[1][1]),
            Keypoint(x=key_pts[2][0], y=key_pts[2][1]),
            Keypoint(x=key_pts[3][0], y=key_pts[3][1])
        ], shape=image.shape)

        seq = iaa.Sequential([
            # iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                rotate=(-30, 30),
                scale=(0.75, 1.25)

            )  # rotate by exactly 10deg and scale to 50-70%, affects keypoints
        ])

        # Augment keypoints and images.
        image_aug, kps_aug = seq(image=image, keypoints=kps)

        key_pts_aug = [[0, 0] for i in range(4)]
        for i in range(len(kps.keypoints)):
            before = kps.keypoints[i]
            after = kps_aug.keypoints[i]
            # print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
            #     i, before.x, before.y, after.x, after.y))
            key_pts_aug[i][0] = after.x
            key_pts_aug[i][1] = after.y

        key_pts_aug = np.array(key_pts_aug)


        # plt.imshow(image_aug)
        # plt.savefig("aug.png")
        # plt.show()

        return {'image': image_aug, 'keypoints': key_pts_aug}





class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image,  'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors【0，1】"""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # if image has no grayscale color channel, add one
        if (len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #heatmaps = heatmaps.transpose((2, 0, 1))
        gt = torch.from_numpy(key_pts)
        gt2 = torch.from_numpy(key_pts)
        return {'image': torch.from_numpy(image),
                #'heatmaps': torch.from_numpy(heatmaps),
                'keypoints': torch.from_numpy(key_pts)}


#tensor图像转为0-1的原始image，结果可视化

def getoriimage(sample):
        image = sample.numpy()
        for i in range(image.shape[0]):
            image[i,:,:] = (image[i,:,:] + 1) * 0.5

        image = image.transpose((1, 2, 0))
        return image



