import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders import make_data_loader
from dataloaders.utils import decode_seg_map_sequence, decode_segmap
from modeling.panoptic_deeplab import PanopticDeepLab
from modeling.sync_batchnorm.replicate import patch_replication_callback
from mypath import Path
from utils.calculate_weights import calculate_weigths_labels
from utils.loss import PanopticLosses, SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.summaries import TensorboardSummary


class Tester(object):
    NUM_CLASSES = 21

    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.nclass,
        ) = make_data_loader(args, **kwargs)

        # Define network
        model = PanopticDeepLab(
            num_classes=self.nclass,
            backbone=args.backbone,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
        )

        self.model = model

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_loader)
        )

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.args.gpu_ids
            )
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        if not os.path.isfile(args.resume):
            raise RuntimeError(
                "=> no checkpoint/model found at '{}'".format(args.resume)
            )
        checkpoint = torch.load(args.resume)
        if args.cuda:
            self.model.module.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint["state_dict"])
        self.best_pred = checkpoint["best_pred"]
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
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

        # in label IDs, according to cityscapes
        self.things_category = [24, 25, 26, 27, 28, 31, 32, 33]

    def resize_tensor(self, tensor):
        # tensor = F.interpolate(
        #     tensor, size=(1024, 2048), mode="nearest",
        # )
        tensor = F.upsample_nearest(
            tensor.unsqueeze(0).float(), size=(1024, 2048)
        )
        return tensor[0]

    def test_and_save(self):
        """Test panoptic segmentation model and save as instanceId images
        """

        self.model.eval()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0
        for i, (sample, filepath) in enumerate(tbar):
            image = sample["image"]

            if self.args.cuda:
                image = image.cuda()

            new_filepath = filepath[0].replace(
                "gtFine_trainvaltest", "gtFine_test_result"
            )
            filename = new_filepath.split("/")[-1]

            print("filepath: ", filepath[0])
            directorypath = new_filepath.split("/")[:-1]
            directorypath = "/".join(directorypath)
            if not os.path.exists(directorypath):
                os.makedirs(directorypath)

            input_image = image.clone()
            with torch.no_grad():
                try:
                    output = self.model(input_image)
                except ValueError as identifier:
                    # there was an error with wrong input size
                    print("Error: ", identifier)
                    continue

            (
                semantic_pred,
                center_pred,
                x_offset_pred,
                y_offset_pred,
            ) = output

            # ############## to create the InstanceIDs
            # See explanations here: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py
            # 1. convert trainIDs to labelIDs
            semantic_labels = np.argmax(semantic_pred.cpu(), axis=1)

            for trainId in range(self.NUM_CLASSES):
                semantic_labels[semantic_labels == trainId] = (
                    self.valid_classes[trainId] + 100
                )
            semantic_labels -= 100  # revert back to original IDs

            # 2. get the instances IDs
            # TODO: resize before here?
            center_pred = center_pred[0]
            x_offset_pred = x_offset_pred[0] / 2
            y_offset_pred = y_offset_pred[0] / 2
            # semantic_labels, center_pred, x_offset_pred, y_offset_pred = map(
            #     self.resize_tensor,
            #     [semantic_labels, center_pred, x_offset_pred, y_offset_pred],
            # )
            print(center_pred.shape)
            plt.show(center_pred[0].cpu().numpy())
            plt.show()
            instances = self.get_instances(
                semantic_labels, center_pred, x_offset_pred, y_offset_pred
            )

            # 4. and add to semantic_labels
            # final_instance_image = semantic_labels_maxed.cuda() + instances
            # final_instance_image = semantic_labels.cuda() + instances

            final_instance_image = torch.where(
                instances == 0,
                semantic_labels.cuda(),
                (semantic_labels.cuda() * 1000) + instances.int(),
            )

            # we want;
            # frankfurt_000000_000294_gtFine_labelIds.png ->
            # with frankfurt_000000_000294_gtFine_instanceIds.png
            save_filepath = (
                directorypath
                + "/"
                + filename.replace("labelIds", "instanceIds")
            )

            # final_instance_image = semantic_labels
            # save instance image (finally!)
            final_instance_image = self.resize_tensor(final_instance_image)
            final_instance_image = (
                final_instance_image[0].cpu().numpy().astype(np.int32)
            )
            # # TODO: remove shows
            # plt.imshow(final_instance_image)
            # plt.show()

            instance_image = Image.fromarray(final_instance_image, "I")
            # instance_image.show()
            instance_image.save(save_filepath)

    def get_instances(
        self, semantic_labels, center, x_offset, y_offset, center_threshold=128
    ):
        mask = torch.zeros_like(semantic_labels)
        for num in self.things_category:
            mask = torch.where(semantic_labels == num, semantic_labels, mask)

        # remove pixels in center that belong to stuffs
        center = center * mask.cuda()

        # 1.0 get center points
        # max pool according to paper
        centers_max = F.max_pool2d(center, 7, stride=1, padding=3)
        centers_select = torch.where(
            center == centers_max, center, torch.zeros_like(center)
        )

        # 2.0 choose top k points, need to sort, then choose the top
        centers_select = torch.where(
            centers_select > center_threshold,
            torch.ones_like(centers_select) * 255,
            torch.zeros_like(centers_select),
        )

        # 3.0 do regressions
        # first make tensors that represent where the centers are for each point
        _, h, w = x_offset.shape
        gridy, gridx = torch.meshgrid(torch.arange(h), torch.arange(w))
        gridx = gridx.cuda().unsqueeze(0)
        gridy = gridy.cuda().unsqueeze(0)

        offsetted_pixloc_x = gridx + x_offset
        offsetted_pixloc_y = gridy + y_offset

        # get indices of center points TODO: ensure correct axis
        center_points = centers_select.nonzero()

        if center_points.shape[0] < 1:
            return torch.zeros_like(semantic_labels[0]).cuda()
        center_points_x = center_points[:, 2:3].unsqueeze(
            -1
        ) * torch.ones_like(x_offset)
        center_points_y = center_points[:, 1:2].unsqueeze(
            -1
        ) * torch.ones_like(x_offset)

        distance_x = (center_points_x - offsetted_pixloc_x) ** 2
        distance_y = (center_points_y - offsetted_pixloc_y) ** 2

        distance_xy = torch.sqrt(distance_x + distance_y)

        instances = (torch.argmin(distance_xy, 0) + 1).float()
        instances = torch.where(
            mask.cuda() == 0, torch.zeros_like(instances), instances,
        )

        # TODO: remove
        # for debug:
        # show_image = centers_select[0].cpu().numpy()
        # plt.imshow(show_image)
        # plt.show()
        # exit(0)
        return instances


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Panoptic Deeplab Training"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet",
        choices=["xception_3stage", "mobilenet_3stage", "resnet_3stage"],
        help="backbone name (default: resnet)",
    )
    parser.add_argument(
        "--out-stride",
        type=int,
        default=16,
        help="network output stride (default: 8)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pascal",
        choices=["pascal", "coco", "cityscapes"],
        help="dataset name (default: pascal)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="segmentation",
        choices=["segmentation", "panoptic"],
        help="training task (default: segmentation)",
    )
    parser.add_argument(
        "--use-sbd",
        action="store_true",
        default=True,
        help="whether to use SBD dataset (default: True)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="dataloader threads",
    )
    parser.add_argument(
        "--base-size", type=int, default=513, help="base image size"
    )
    parser.add_argument(
        "--crop-size", type=int, default=513, help="crop image size"
    )
    parser.add_argument(
        "--sync-bn",
        type=bool,
        default=None,
        help="whether to use sync bn (default: auto)",
    )
    parser.add_argument(
        "--freeze-bn",
        type=bool,
        default=False,
        help="whether to freeze bn parameters (default: False)",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="loss func type (default: ce)",
    )
    # training hyper params
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="number of epochs to train (default: auto)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        metavar="N",
        help="start epochs (default:0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="input batch size for \
                                training (default: auto)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="input batch size for \
                                testing (default: auto)",
    )
    parser.add_argument(
        "--use-balanced-weights",
        action="store_true",
        default=False,
        help="whether to use balanced weights (default: False)",
    )
    # optimizer params
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (default: auto)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="poly",
        choices=["poly", "step", "cos"],
        help="lr scheduler mode: (default: poly)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        metavar="M",
        help="w-decay (default: 5e-4)",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=False,
        help="whether use nesterov (default: False)",
    )
    # cuda, seed and logging
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    # checking point
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="put the path to resuming file if needed",
    )
    parser.add_argument(
        "--checkname", type=str, default=None, help="set the checkpoint name"
    )
    # finetuning pre-trained models
    parser.add_argument(
        "--ft",
        action="store_true",
        default=False,
        help="finetuning on a different dataset",
    )
    # evaluation option
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="evaluation interval (default: 1)",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        default=False,
        help="skip validation during training",
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        except ValueError:
            raise ValueError(
                "Argument --gpu_ids must be a comma-separated list of integers only"
            )

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.test_batch_size is None:
        args.test_batch_size = 1

    print(args)
    torch.manual_seed(args.seed)
    tester = Tester(args)
    tester.test_and_save()


if __name__ == "__main__":
    main()
