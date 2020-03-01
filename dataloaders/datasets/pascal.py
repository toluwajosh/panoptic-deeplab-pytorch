from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

try:
    from .read_from_xml import parse_object_bbox
    from .make_gaussian import make_gaussian
except ModuleNotFoundError as identifier:
    from read_from_xml import parse_object_bbox
    from make_gaussian import make_gaussian


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    NUM_CLASSES = 21

    def __init__(
        self, args, base_dir=Path.db_root_dir("pascal"), split="train",
    ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, "JPEGImages")
        self._cat_dir = os.path.join(self._base_dir, "SegmentationClass")
        self._annot_dir = os.path.join(self._base_dir, "Annotations")

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, "ImageSets", "Segmentation")

        self.im_ids = []
        self.images = []
        self.categories = []
        self.annotations = []

        for splt in self.split:
            with open(
                os.path.join(os.path.join(_splits_dir, splt + ".txt")), "r"
            ) as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                _annot = os.path.join(self._annot_dir, line + ".xml")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.annotations.append(_annot)

        assert len(self.images) == len(self.categories)

        # Display stats
        print("Number of images in {}: {:d}".format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {
            "image": _img,
            "label": _target,
        }

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == "val":
                return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])

        return _img, _target

    def _make_data_set(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])

        _centers_image, x_reg, y_reg = self.load_centers_and_regression(
            self.annotations[index], _img.size
        )

        return _img, _target, _centers_image, x_reg, y_reg

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                ),
                tr.RandomGaussianBlur(),
                tr.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose(
            [
                tr.FixScaleCrop(crop_size=self.args.crop_size),
                tr.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def __str__(self):
        return "VOC2012(split=" + str(self.split) + ")"


class VOCPanoptic(Dataset):
    """
    PascalVoc dataset
    """

    NUM_CLASSES = 21

    def __init__(
        self, args, base_dir=Path.db_root_dir("pascal"), split="train",
    ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, "JPEGImages")
        self._cat_dir = os.path.join(self._base_dir, "SegmentationClass")
        self._annot_dir = os.path.join(self._base_dir, "Annotations")

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, "ImageSets", "Segmentation")

        self.im_ids = []
        self.images = []
        self.categories = []
        self.annotations = []

        for splt in self.split:
            with open(
                os.path.join(os.path.join(_splits_dir, splt + ".txt")), "r"
            ) as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                _annot = os.path.join(self._annot_dir, line + ".xml")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.annotations.append(_annot)

        assert len(self.images) == len(self.categories)

        # Display stats
        print("Number of images in {}: {:d}".format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target, _centers, x_reg, y_reg = self._make_data_set(index)
        _centers = Image.fromarray(np.uint8(_centers * 255))
        x_reg = Image.fromarray(np.int32(x_reg), "I")
        y_reg = Image.fromarray(np.int32(y_reg), "I")

        sample = {
            "image": _img,
            "label": _target,
            "center": _centers,
            "x_reg": x_reg,
            "y_reg": y_reg,
        }

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == "val":
                return self.transform_val(sample)

    def load_centers_and_regression(self, annotation_file, size):
        annotation_bbox = parse_object_bbox(annotation_file)
        # we need to know the size of image
        # centers_image = np.ones([size[1], size[0]])
        centers_image = np.zeros([size[1], size[0]])
        x_reg = np.zeros([size[1], size[0]])
        y_reg = np.zeros([size[1], size[0]])
        for center in annotation_bbox:
            # replace path with 2d gaussian
            x0 = int(center["xmin"])
            x1 = int(center["xmax"])
            y0 = int(center["ymin"])
            y1 = int(center["ymax"])
            if (x1 - x0) % 2 != 0:
                x1 -= 1
            if (y1 - y0) % 2 != 0:
                y1 -= 1

            w = x1 - x0
            h = y1 - y0

            c_x = w // 2
            c_y = h // 2

            gaussian_patch = make_gaussian([w, h], 8)
            # centers_image[y0:y1, x0:x1] -= gaussian_patch
            centers_image[y0:y1, x0:x1] = np.maximum(
                centers_image[y0:y1, x0:x1], gaussian_patch
            )

            x_patch = np.tile(np.arange(-c_x, c_x), (h, 1))
            y_patch = np.tile(np.arange(-c_y, c_y), (w, 1)).T

            # x_reg[y0:y1, x0:x1] = np.maximum(x_reg[y0:y1, x0:x1], x_patch)
            # y_reg[y0:y1, x0:x1] = np.maximum(y_reg[y0:y1, x0:x1], y_patch)
            x_reg[y0:y1, x0:x1] = x_patch
            y_reg[y0:y1, x0:x1] = y_patch
        return centers_image, x_reg, y_reg

    def _make_data_set(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])

        _centers_image, x_reg, y_reg = self.load_centers_and_regression(
            self.annotations[index], _img.size
        )

        return _img, _target, _centers_image, x_reg, y_reg

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                ),
                tr.RandomGaussianBlur(),
                tr.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose(
            [
                tr.FixScaleCrop(crop_size=self.args.crop_size),
                tr.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def __str__(self):
        return "VOC2012(split=" + str(self.split) + ")"


if __name__ == "__main__":
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCPanoptic(args, split="train")

    dataloader = DataLoader(
        voc_train, batch_size=1, shuffle=True, num_workers=0
    )

    for ii, sample in enumerate(dataloader, 1):
        for jj in range(sample["image"].size()[0]):
            # print(sample.keys())
            # exit(0)
            img = sample["image"].numpy()
            gt = sample["label"].numpy()
            # mask = np.zeros_like(gt)
            # mask[gt > 0] = 1
            # cen = mask[0]

            center = sample["center"].numpy()[0]
            x_reg = sample["x_reg"].numpy()[0]
            y_reg = sample["y_reg"].numpy()[0]

            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset="pascal")
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            center = center.astype(np.uint8) / 255.0

            print(np.max(center))
            x_reg = x_reg.astype(np.uint8)
            y_reg = y_reg.astype(np.uint8)

            plt.figure()
            plt.title("display")
            plt.subplot(311)
            plt.imshow(img_tmp)
            plt.subplot(312)
            plt.imshow(segmap)
            plt.subplot(313)
            plt.imshow(center)
            plt.subplot(321)
            plt.imshow(x_reg)
            # plt.subplot(522)
            # plt.imshow(y_reg)

        if ii == 1:
            break

    plt.show(block=True)
