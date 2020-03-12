import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import cv2


try:
    from .read_from_json import load_json_data
    from .make_gaussian import make_gaussian
except ModuleNotFoundError as identifier:
    from read_from_json import load_json_data
    from make_gaussian import make_gaussian


class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19

    def __init__(
        self, args, root=Path.db_root_dir("cityscapes"), split="train"
    ):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine_trainvaltest", "gtFine", self.split
        )

        self.files[split] = self.recursive_glob(
            rootdir=self.images_base, suffix=".png"
        )

        self.void_classes = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            9,
            10,
            14,
            15,
            16,
            18,
            29,
            30,
            -1,
        ]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s"
                % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        _img = Image.open(img_path).convert("RGB")
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {"image": _img, "label": _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == "val":
            return self.transform_val(sample)
        elif self.split == "test":
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir=".", suffix=""):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                    fill=255,
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

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose(
            [
                tr.FixedResize(size=self.args.crop_size),
                tr.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)


class CityscapesPanoptic(data.Dataset):
    NUM_CLASSES = 21

    def __init__(
        self, args, root=Path.db_root_dir("cityscapes"), split="train"
    ):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.annotations = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine_trainvaltest", "gtFine", self.split
        )

        self.files[split] = self.recursive_glob(
            rootdir=self.images_base, suffix=".png"
        )
        self.annotations[split] = self.recursive_glob(
            rootdir=self.images_base, suffix=".json"
        )

        self.void_classes = [
            # 0,
            1,
            2,
            3,
            4,
            5,
            6,
            9,
            10,
            14,
            15,
            16,
            18,
            29,
            # 30,
            -1,
        ]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
            30,  # added
            0,  # added
        ]
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "trailer",  # added
            "unlabelled",  # added
        ]

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        # hardcoded things category
        # self.things_category = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
        self.things_category = [
            "pole",
            "traffic light",
            "traffic sign",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "trailer",
        ]

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s"
                % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def load_centers_and_regression(self, annotation_file, size):
        annotation_data = load_json_data(annotation_file)

        # we need to know the size of image
        # centers_image = np.ones([size[1], size[0]])
        centers_image = np.zeros([size[1], size[0]])
        x_reg = np.zeros([size[1], size[0]])
        y_reg = np.zeros([size[1], size[0]])
        for object_data in annotation_data:
            center = object_data["bbox"]
            label = object_data["label"]
            if label not in self.things_category:
                continue
            polygon = np.int0(object_data["polygon"])
            minx = np.min(polygon[:, 0])
            miny = np.min(polygon[:, 1])

            x, y, w, h = cv2.boundingRect(polygon)

            x0 = max(x, 0)
            x1 = min(x + w, size[0])
            y0 = max(y, 0)
            y1 = min(y + h, size[1])

            if (x1 - x0) % 2 != 0:
                x1 -= 1
            if (y1 - y0) % 2 != 0:
                y1 -= 1
            w = x1 - x0
            h = y1 - y0

            c_x = w // 2
            c_y = h // 2
            gaussian_patch = make_gaussian([w, h], center=[c_x, c_y])

            mask = np.zeros_like(gaussian_patch)

            # adjust polygon coordinates
            polygon[:, 0] = polygon[:, 0] - minx
            polygon[:, 1] = polygon[:, 1] - miny
            cv2.fillPoly(mask, pts=[polygon], color=(1, 1, 1))

            try:
                pass
                centers_image[y0:y1, x0:x1] = np.maximum(
                    centers_image[y0:y1, x0:x1], gaussian_patch
                )
                # centers_image[y0:y1, x0:x1] = np.where(
                # mask == 1, gaussian_patch, centers_image[y0:y1, x0:x1]
                # )
            except ValueError as identifier:
                print("\n")
                print(identifier)
                print(
                    "w: {} h: {} x0: {} x1: {} y0: {} y1: {}".format(
                        w, h, x0, x1, y0, y1
                    )
                )
                print(centers_image.shape)
                print(centers_image[y0:y1, x0:x1].shape)
                print(gaussian_patch.shape)
                raise

            x_patch = np.tile(np.arange(-c_x, c_x), (h, 1))
            y_patch = np.tile(np.arange(-c_y, c_y), (w, 1)).T
            x_reg[y0:y1, x0:x1] = np.where(
                mask == 1, x_patch, x_reg[y0:y1, x0:x1]
            )
            y_reg[y0:y1, x0:x1] = np.where(
                mask == 1, y_patch, y_reg[y0:y1, x0:x1]
            )
        return centers_image, x_reg, y_reg

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        _img = Image.open(img_path).convert("RGB")
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        # centers, x_reg, y_reg
        annotation_file = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_polygons.json",
        )
        _centers, x_reg, y_reg = self.load_centers_and_regression(
            annotation_file, _img.size
        )
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

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == "val":
            return self.transform_val(sample)
        elif self.split == "test":
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir=".", suffix=""):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                    fill=255,
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

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose(
            [
                # tr.FixedResize(size=self.args.crop_size),
                tr.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)


if __name__ == "__main__":
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesPanoptic(args, split="train")

    dataloader = DataLoader(
        cityscapes_train, batch_size=1, shuffle=True, num_workers=2
    )

    for ii, sample in enumerate(dataloader):
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

            print(img.shape)
            print(np.max(x_reg))
            print(np.min(x_reg))
            print(np.max(y_reg))
            print(np.min(y_reg))
            # exit(0)

            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset="pascal")
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            center = center.astype(np.uint8) / 255.0

            plt.figure()
            plt.title("display")
            # plt.subplot(221)
            # plt.imshow(img_tmp)
            plt.subplot(221)
            plt.imshow(segmap)
            plt.subplot(222)
            plt.imshow(center)
            plt.subplot(223)
            plt.imshow(x_reg)
            plt.subplot(224)
            plt.imshow(y_reg)

        if ii == 0:
            break

    plt.show(block=True)

